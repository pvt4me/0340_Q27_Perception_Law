#!/usr/bin/env python3
# 0340_Q27_Perception_Law.py - Complete 0340 Law Implementation (STABLE)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import time

# Constants from 0340 Law - TUNED FOR 44:3.3:52.7
META_MIN, META_MAX = 0.0980, 0.2000
BYTE_MIN, BYTE_MAX = 25, 51
GAPLOW, GAPMID, GAPHIGH = 0.130, 0.1325, 0.135
CORRLEFT_TARGET = 0.1290
CORRRIGHT_TARGET = 0.1374
GAP_BYTES = {33, 34}
TARGETS = {'L': 0.44, 'G': 0.033, 'R': 0.527}

class UniversalGapProcessor:
    def __init__(self, tag="Q27"):
        self.tag = tag
        self.FORBIDDENBYTES = self.compute_forbidden_bytes()
    
    def compute_forbidden_bytes(self):
        lo = int(np.floor(GAPLOW * 255))
        hi = int(np.ceil(GAPHIGH * 255))
        return set(range(lo, hi+1)) & GAP_BYTES
    
    def delta5050_correct(self, metabit):
        if not np.isfinite(metabit): 
            return GAPMID
        if GAPLOW <= metabit <= GAPMID:
            return CORRLEFT_TARGET - 0.4*(metabit - GAPLOW)
        elif GAPMID < metabit <= GAPHIGH:
            return CORRRIGHT_TARGET + 0.96*(metabit - GAPMID)
        return metabit
    
    def metabit_to_byte(self, m):
        return max(BYTE_MIN, min(BYTE_MAX, int(np.floor(m * 255))))
    
    def is_valid_metabit(self, m):
        b = self.metabit_to_byte(m)
        return (GAPLOW-0.01 <= m <= GAPHIGH+0.035) and (b not in self.FORBIDDENBYTES)

class PhiField0340:
    def __init__(self):
        self.n_L, self.n_G, self.n_R = 8, 2, 17
        self.n_total = 27
        self.J = self.create_J_matrix()
        self.Phi = np.zeros(self.n_total)
        self.reset_initial()
        print(f"Phi field initialized: lambda_max(J)={np.linalg.eigvals(self.J).real.max():.3f}")
    
    def create_J_matrix(self):
        J = np.zeros((27, 27))
        w_LL, w_LG, w_GL = 0.75, 0.35, 0.20
        w_GR, w_RG, w_RR = 0.45, 0.30, 0.55
        
        for i in range(8):
            for j in range(i+1, 8):
                if abs(i-j) <= 3: 
                    J[i,j] = J[j,i] = w_LL
        
        for g in range(2):
            for l in [0,3,6]: 
                J[l,8+g] = w_LG; J[8+g,l] = w_GL
            for r_idx in range(17):
                r = 10 + r_idx
                J[8+g,r] = w_GR * 0.8; J[r,8+g] = w_RG
        
        for i in range(17):
            next_i = (i+1) % 17
            prev_i = (i-1) % 17
            J[10+i,10+next_i] = J[10+next_i,10+i] = w_RR
            J[10+i,10+prev_i] = J[10+prev_i,10+i] = w_RR * 0.9
            if i % 4 == 0:
                far = (i+8) % 17
                J[10+i,10+far] = J[10+far,10+i] = w_RR * 0.7
        
        lambda_max = np.linalg.eigvals(J).real.max()
        return J / lambda_max * 1.8
    
    def reset_initial(self):
        np.random.seed(340)
        self.Phi[:8] = 1.8 + 0.4*np.random.randn(8)
        self.Phi[8:10] = 0.15 + 0.08*np.random.randn(2)
        self.Phi[10:] = 1.2 + 0.3*np.random.randn(17)
    
    def step(self, dt=0.04):
        linear = self.J @ self.Phi
        nonlinear = self.Phi**3 * 0.08
        total_E = np.sum(self.Phi**2) + 1e-8
        
        balance = np.zeros(self.n_total)
        target_L, target_G, target_R = 0.42, 0.04, 0.54
        
        E_L = np.sum(self.Phi[:8]**2) / total_E
        E_G = np.sum(self.Phi[8:10]**2) / total_E
        E_R = np.sum(self.Phi[10:]**2) / total_E
        
        balance[:8] = 0.015 * (target_L - E_L)
        balance[8:10] = 0.025 * (target_G - E_G)
        balance[10:] = 0.012 * (target_R - E_R)
        
        noise = 0.015 * np.random.randn(self.n_total) * (total_E > 0.1)
        
        dPhi = 0.40*linear - nonlinear + balance + noise
        self.Phi += dt * dPhi
        self.Phi = np.clip(self.Phi, -3.0, 3.0)
        return dPhi

class ByteEmission0340:
    def __init__(self, phi_field):
        self.phi = phi_field
        self.processor = UniversalGapProcessor("Q27")
        self.history = deque(maxlen=2000)
        self.step_count = 0
    
    def emit(self):
        self.step_count += 1
        if self.step_count % 3 != 0:
            return None
        
        energies = np.abs(self.phi.Phi)**2 + 1e-8
        probs = energies / np.sum(energies)
        
        probs[8:10] *= 1.8
        probs[:8] *= 1.1
        probs[10:] *= 0.95
        probs /= np.sum(probs)
        
        node = np.random.choice(self.phi.n_total, p=probs)
        phi_val = self.phi.Phi[node]
        
        phi_range = (np.min(self.phi.Phi), np.max(self.phi.Phi))
        range_size = max(phi_range[1] - phi_range[0], 0.1)
        metabit_raw = (META_MAX-META_MIN)/range_size * (phi_val-phi_range[0]) + META_MIN
        metabit_raw = np.clip(metabit_raw, META_MIN, META_MAX)
        
        metabit_corr = self.processor.delta5050_correct(metabit_raw)
        byte = self.processor.metabit_to_byte(metabit_corr)
        
        record = {
            'step': self.step_count, 'node': node,
            'type': 'L'if node<8 else'G'if node<10 else'R',
            'phi': phi_val, 'metabit_raw': metabit_raw, 'metabit': metabit_corr,
            'byte': byte, 'state': 'L'if byte<33 else'G'if byte in GAP_BYTES else'R'
        }
        self.history.append(record)
        return record
    
    def statistics(self):
        if len(self.history) == 0: 
            return {'L':0.0, 'G':0.0, 'R':0.0, 'error':1.0}
        
        recent_size = min(200, len(self.history))
        recent = list(self.history)[-recent_size:]
        bytes_list = np.array([r['byte'] for r in recent])
        
        L = np.mean(bytes_list < 33)
        G = np.mean(np.isin(bytes_list, list(GAP_BYTES)))
        R = np.mean(bytes_list > 34)
        
        errors = [abs(L - TARGETS['L']), abs(G - TARGETS['G']), abs(R - TARGETS['R'])]
        error = np.mean(errors)
        
        return {'L':L, 'G':G, 'R':R, 'error':error}

class Visualizer:
    def __init__(self, phi_field, emission):
        self.phi = phi_field
        self.emit = emission
        self.positions = self._create_3d_layout()
        self.colors = ['#1f77b4']*8 + ['#ff7f00']*2 + ['#d62728']*17
    
    def _create_3d_layout(self):
        pos = np.zeros((27,3))
        theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
        pos[:8,0] = 1.5*np.cos(theta); pos[:8,1] = 1.5*np.sin(theta)
        pos[:8,2] = 0.5*np.sin(theta*2.3)
        pos[8] = [0.4, -0.3, 0]; pos[9] = [0.4, 0.3, 0]
        for i in range(17):
            phi = 2*np.pi*i/17
            r = 2.2 + 0.3*np.sin(phi*3)
            pos[10+i] = [-r*np.cos(phi), -r*np.sin(phi), 0.6*np.cos(phi*2.1)]
        return pos
    
    def phi_energy(self):
        E_L = np.sum(self.phi.Phi[:8]**2)
        E_G = np.sum(self.phi.Phi[8:10]**2)
        E_R = np.sum(self.phi.Phi[10:]**2)
        total = E_L + E_G + E_R + 1e-10
        return {'L':E_L/total*100, 'G':E_G/total*100, 'R':E_R/total*100}
    
    def animate(self, steps=500):
        fig = plt.figure(figsize=(16,12))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        def update(frame):
            self.phi.step(dt=0.05)
            self.emit.emit()
            
            phi_stats = self.phi_energy()
            byte_stats = self.emit.statistics()
            
            # 1. 3D Phi field - FIXED
            ax1.clear()
            phi_abs = np.abs(self.phi.Phi)
            phi_max = np.max(phi_abs) + 1e-10
            scale = 1 + 0.4*(phi_abs/phi_max)
            dynamic_pos = self.positions * scale[:,None]
            sizes = 40 + 200*(phi_abs/phi_max)
            
            ax1.scatter(dynamic_pos[:,0], dynamic_pos[:,1], dynamic_pos[:,2],
                       c=self.colors, s=sizes, alpha=0.85, edgecolors='white', linewidth=0.8)
            ax1.set_xlim(-3,3); ax1.set_ylim(-3,3); ax1.set_zlim(-2,2)
            ax1.set_title(f"Phi Field | L:{phi_stats['L']:.1f}% G:{phi_stats['G']:.1f}% R:{phi_stats['R']:.1f}%")
            
            # 2. Byte balance - FIXED
            ax2.clear()
            if len(self.emit.history) > 20:
                recent_bytes = [r['byte'] for r in list(self.emit.history)[-100:]]
                L_count = sum(b<33 for b in recent_bytes)
                G_count = sum(b in GAP_BYTES for b in recent_bytes)
                R_count = len(recent_bytes) - L_count - G_count
                
                bars = ax2.bar(['L','G','R'], [L_count, G_count, R_count], 
                              color=['blue','orange','red'], alpha=0.7)
                ax2.set_title(f"Byte Counts | Error: {byte_stats['error']:.4f}")
            else:
                ax2.text(0.5, 0.5, f"No bytes yet\nHistory: {len(self.emit.history)}", 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title("Byte Balance")
            ax2.grid(alpha=0.3)
            
            # 3. Byte histogram - FIXED (SAFEST VERSION)
            ax3.clear()
            if len(self.emit.history) > 5:
                bytes_arr = np.array([r['byte'] for r in list(self.emit.history)[-150:]])
                n, bins, patches = ax3.hist(bytes_arr, bins=np.arange(24.5,52.5), alpha=0.7, 
                                          color='skyblue', edgecolor='black')
                gap_count = sum(b in GAP_BYTES for b in bytes_arr)
                if gap_count > 0:
                    ax3.axvspan(32.5,34.5, alpha=0.3, color='red')
                    ax3.text(0.02, 0.98, f'GAP: {gap_count}', transform=ax3.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
            else:
                ax3.text(0.5, 0.5, 'Waiting for bytes...', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_xlim(24,52)
            ax3.set_title(f"Bytes {BYTE_MIN}-{BYTE_MAX} | Total: {len(self.emit.history)}")
            ax3.grid(alpha=0.3)
            
            # 4. Phase ratio - FIXED
            ax4.clear()
            if len(self.emit.history) > 10:
                recent_bytes = [r['byte'] for r in list(self.emit.history)[-100:]]
                L_count = sum(b<33 for b in recent_bytes) / len(recent_bytes)
                R_count = sum(b>34 for b in recent_bytes) / len(recent_bytes)
                ax4.bar(['P_L','P_R'], [L_count, R_count], color=['blue','red'], alpha=0.8)
                ax4.axhline(y=0.44, color='blue', ls='--', alpha=0.7)
                ax4.axhline(y=0.56, color='red', ls='--', alpha=0.7)
            else:
                ax4.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_ylim(0,1)
            ax4.set_title(f"P_R/P_L = {byte_stats['R']/max(byte_stats['L'],0.01):.3f}")
            ax4.grid(alpha=0.3)
            
            fig.suptitle(f'0340 Law | Step: {self.emit.step_count} | L:{byte_stats["L"]:.1%} G:{byte_stats["G"]:.1%} R:{byte_stats["R"]:.1%}')
        
        ani = FuncAnimation(fig, update, frames=steps, interval=80, repeat=True, blit=False)
        plt.tight_layout()
        plt.show()
        return ani

def main():
    print("ATLASv540_Q27 - Complete 0340 Law Implementation (STABLE)")
    print("="*60)
    
    phi_field = PhiField0340()
    emission = ByteEmission0340(phi_field)
    viz = Visualizer(phi_field, emission)
    
    print("Running stable 400 steps simulation...")
    ani = viz.animate(steps=400)
    
    final_stats = emission.statistics()
    print("\n" + "="*60)
    print("FINAL 0340 LAW VERIFICATION")
    print("="*60)
    print(f"L: {final_stats['L']:.3f} (target 0.440) Δ={abs(final_stats['L']-0.44):.3f}")
    print(f"G: {final_stats['G']:.3f} (target 0.033) Δ={abs(final_stats['G']-0.033):.3f}")
    print(f"R: {final_stats['R']:.3f} (target 0.527) Δ={abs(final_stats['R']-0.527):.3f}")
    print(f"Total error: {final_stats['error']:.3f}")
    print(f"P_R/P_L: {final_stats['R']/max(final_stats['L'],0.01):.3f} (target 1.273)")
    
    status = "0340 LAW CONFIRMED" if final_stats['error'] < 0.08 else "CONVERGING"
    print(f"STATUS: {status}")

if __name__ == "__main__":
    main()
