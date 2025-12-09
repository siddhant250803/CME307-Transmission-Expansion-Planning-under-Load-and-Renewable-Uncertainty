"""
Enhanced Visualization Module for TEP Project
==============================================

This module provides publication-quality visualizations for the 
Transmission Expansion Planning (TEP) project. All plots use a 
consistent aesthetic theme suitable for academic reports.

Author: CME307 Team (Edouard Rabasse, Siddhant Sukhani)
Date: December 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# THEME CONFIGURATION
# =============================================================================

# Custom color palette - inspired by energy/power systems
COLORS = {
    'primary': '#1E3A5F',       # Deep navy blue
    'secondary': '#3C6E71',     # Teal green
    'accent': '#D9534F',        # Coral red
    'warning': '#F0AD4E',       # Amber orange
    'success': '#5CB85C',       # Green
    'light': '#ECF0F1',         # Light gray
    'dark': '#2C3E50',          # Dark slate
    'grid': '#BDC3C7',          # Grid gray
}

# Generation type colors
GEN_COLORS = {
    'CT': '#FF6B6B',            # Combustion Turbine - coral
    'CC': '#4ECDC4',            # Combined Cycle - teal
    'STEAM': '#95A5A6',         # Steam - gray
    'NUCLEAR': '#9B59B6',       # Nuclear - purple
    'HYDRO': '#3498DB',         # Hydro - blue
    'WIND': '#2ECC71',          # Wind - green
    'PV': '#F39C12',            # Solar PV - golden
    'RTPV': '#E67E22',          # Rooftop PV - orange
    'CSP': '#E74C3C',           # Concentrated Solar - red
    'SYNC_COND': '#7F8C8D',     # Synchronous Condenser - dark gray
}

# Set global style with BIGGER, BOLDER text
def set_publication_style():
    """Configure matplotlib for publication-quality plots with large, bold text."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 14,                    # Base font size
        'font.weight': 'bold',              # Default to bold
        'axes.titlesize': 20,               # Title size
        'axes.titleweight': 'bold',         # Bold titles
        'axes.labelsize': 16,               # Axis label size
        'axes.labelweight': 'bold',         # Bold axis labels
        'xtick.labelsize': 13,              # Tick label size
        'ytick.labelsize': 13,              # Tick label size
        'legend.fontsize': 13,              # Legend font size
        'legend.title_fontsize': 14,        # Legend title size
        'figure.titlesize': 22,             # Figure title size
        'figure.titleweight': 'bold',       # Bold figure titles
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
        'axes.linewidth': 1.5,              # Axis lines
    })


# =============================================================================
# NETWORK VISUALIZATION
# =============================================================================

def plot_network_enhanced(data_loader, results: Optional[Dict] = None, 
                          save_path: Optional[str] = None,
                          title: str = "RTS-GMLC Network Topology",
                          highlight_congested: bool = True) -> plt.Figure:
    """Create a stunning, detailed network visualization."""
    set_publication_style()
    
    buses = data_loader.get_bus_data()
    branches = data_loader.get_branch_data()
    generators = data_loader.get_generator_data()
    
    # Calculate generation capacity per bus
    gen_capacity = {}
    for _, gen in generators.iterrows():
        bus_id = gen['Bus ID']
        gen_capacity[bus_id] = gen_capacity.get(bus_id, 0) + gen['PMax MW']
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(20, 14), facecolor='white')
    ax.set_facecolor('#F8F9FA')
    
    # Create graph
    G = nx.Graph()
    pos = {}
    
    for bus_id in buses.index:
        G.add_node(bus_id)
        if 'lat' in buses.columns and 'lng' in buses.columns:
            pos[bus_id] = (buses.loc[bus_id, 'lng'], buses.loc[bus_id, 'lat'])
    
    # Process edges with flow data for heatmap
    edge_data = []
    for _, branch in branches.iterrows():
        from_bus = int(branch['From Bus'])
        to_bus = int(branch['To Bus'])
        rating = branch['Cont Rating']
        uid = branch['UID']
        
        G.add_edge(from_bus, to_bus, rating=rating, uid=uid)
        
        flow = 0
        utilization = 0
        if results and 'flows' in results:
            flow = abs(results['flows'].get(uid, 0))
            utilization = flow / rating if rating > 0 else 0
        
        edge_data.append({
            'edge': (from_bus, to_bus),
            'uid': uid,
            'rating': rating,
            'flow': flow,
            'utilization': utilization
        })
    
    # Sort edges by utilization for drawing order (low util first, high util on top)
    edge_data.sort(key=lambda x: x['utilization'])
    
    # Create colormap for edges (green -> yellow -> red)
    cmap = plt.cm.RdYlGn_r
    
    # Draw edges with heatmap colors
    for ed in edge_data:
        color = cmap(ed['utilization'])
        width = 1.5 + ed['utilization'] * 6
        alpha = 0.5 + ed['utilization'] * 0.4
        
        x = [pos[ed['edge'][0]][0], pos[ed['edge'][1]][0]]
        y = [pos[ed['edge'][0]][1], pos[ed['edge'][1]][1]]
        ax.plot(x, y, color=color, linewidth=width, alpha=alpha, solid_capstyle='round', zorder=1)
    
    # Draw new transmission lines from robust TEP (thick, bold)
    new_lines = [(101, 106), (101, 117)]
    for from_bus, to_bus in new_lines:
        if from_bus in pos and to_bus in pos:
            x = [pos[from_bus][0], pos[to_bus][0]]
            y = [pos[from_bus][1], pos[to_bus][1]]
            # Shadow effect
            ax.plot(x, y, color='#2C3E50', linewidth=14, alpha=0.3, solid_capstyle='round', zorder=2)
            ax.plot(x, y, color='#E74C3C', linewidth=10, alpha=0.8, solid_capstyle='round', zorder=3)
            ax.plot(x, y, color='#C0392B', linewidth=6, alpha=1.0, solid_capstyle='round', zorder=4)
    
    # Node styling - bolder colors for white background
    region_colors = {1: '#3498DB', 2: '#E74C3C', 3: '#27AE60'}
    
    for bus_id in buses.index:
        region = buses.loc[bus_id, 'Area']
        load = buses.loc[bus_id, 'MW Load']
        cap = gen_capacity.get(bus_id, 0)
        
        # Size based on load (larger = more load)
        size = max(120, min(900, load * 1.8))
        
        # Color by region
        color = region_colors.get(region, '#95A5A6')
        
        # Main node with dark edge
        ax.scatter(pos[bus_id][0], pos[bus_id][1], s=size, c=color, 
                  edgecolors='#2C3E50', linewidths=2, alpha=0.9, zorder=6)
        
        # Generation capacity ring (if has generation)
        if cap > 100:
            ring_size = size * 2
            ax.scatter(pos[bus_id][0], pos[bus_id][1], s=ring_size, facecolors='none',
                      edgecolors='#F39C12', linewidths=3, alpha=0.8, zorder=5)
    
    # Highlight key buses (101, 106, 117) with labels
    key_buses = {101: 'Hub 101\n(New Lines)', 106: 'Bus 106', 117: 'Bus 117'}
    for bus_id, label in key_buses.items():
        if bus_id in pos:
            # Highlight ring
            ax.scatter(pos[bus_id][0], pos[bus_id][1], s=1500, facecolors='none',
                      edgecolors='#F39C12', linewidths=5, alpha=0.95, zorder=7)
            # Label with larger font
            ax.annotate(label, pos[bus_id], xytext=(12, 12), textcoords='offset points',
                       fontsize=14, fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#C0392B', 
                                edgecolor='#2C3E50', alpha=0.95, linewidth=2),
                       zorder=10)
    
    # Label high-load buses with larger font
    for bus_id in buses.index:
        load = buses.loc[bus_id, 'MW Load']
        if load > 300 and bus_id not in key_buses:
            ax.annotate(str(bus_id), pos[bus_id], fontsize=11, fontweight='bold',
                       color='#2C3E50', ha='center', va='center', zorder=8)
    
    # Info box only
    info_text = (
        "RTS-GMLC Power System\n"
        "━━━━━━━━━━━━━━━━━━━━━━━\n"
        "73 Buses | 120 Lines | 158 Generators\n"
        "Total Load: 8,550 MW\n"
        "Congested: A27, CA-1, CB-1\n"
        "New Lines: 101→106, 101→117"
    )
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=14,
           verticalalignment='bottom', horizontalalignment='right',
           fontfamily='monospace', color='#2C3E50', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                    edgecolor='#F39C12', alpha=0.95, linewidth=3))
    
    # Remove axes, no title
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Network plot saved to {save_path}")
    
    return fig


# =============================================================================
# GENERATION MIX VISUALIZATION
# =============================================================================

def plot_generation_mix_enhanced(results: Dict, data_loader, 
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Create an enhanced generation mix visualization using horizontal bars.
    
    This visualization shows:
    - Generation by fuel type as horizontal bars
    - Percentage contribution
    - Capacity factor indication
    
    Parameters
    ----------
    results : dict
        Results dictionary containing 'generation' data
    data_loader : RTSDataLoader
        Data loader instance
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    set_publication_style()
    
    if 'generation' not in results:
        print("No generation data available")
        return None
    
    generators = data_loader.get_generator_data()
    
    # Aggregate by fuel type
    gen_by_type = {}
    capacity_by_type = {}
    
    for gen_id, power in results['generation'].items():
        gen_row = generators[generators['GEN UID'] == gen_id]
        if len(gen_row) > 0:
            gen_type = gen_row.iloc[0]['Unit Type']
            gen_by_type[gen_type] = gen_by_type.get(gen_type, 0) + power
            capacity_by_type[gen_type] = capacity_by_type.get(gen_type, 0) + gen_row.iloc[0]['PMax MW']
    
    if not gen_by_type:
        print("No generation data to plot")
        return None
    
    # Sort by generation amount
    sorted_types = sorted(gen_by_type.keys(), key=lambda x: gen_by_type[x], reverse=True)
    
    # Filter out zero/negligible generation
    sorted_types = [t for t in sorted_types if gen_by_type[t] > 1]
    
    powers = [gen_by_type[t] for t in sorted_types]
    capacities = [capacity_by_type[t] for t in sorted_types]
    cap_factors = [p/c*100 if c > 0 else 0 for p, c in zip(powers, capacities)]
    
    total_gen = sum(powers)
    percentages = [p/total_gen*100 for p in powers]
    
    colors = [GEN_COLORS.get(t, '#95A5A6') for t in sorted_types]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Generation by type (horizontal bars)
    y_pos = np.arange(len(sorted_types))
    bars = ax1.barh(y_pos, powers, color=colors, edgecolor=COLORS['dark'], linewidth=1.0)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_types, fontsize=14, fontweight='bold')
    ax1.set_xlabel('Generation (MW)', fontsize=16, fontweight='bold')
    ax1.set_title('Generation by Fuel Type', fontsize=20, fontweight='bold')
    ax1.invert_yaxis()
    ax1.tick_params(axis='x', labelsize=13)
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        width = bar.get_width()
        ax1.text(width + 50, bar.get_y() + bar.get_height()/2,
                f'{width:,.0f} MW ({pct:.1f}%)',
                va='center', ha='left', fontsize=12, fontweight='bold', color=COLORS['dark'])
    
    ax1.set_xlim(0, max(powers) * 1.4)
    ax1.axvline(x=0, color=COLORS['dark'], linewidth=1.0)
    
    # Right plot: Capacity factor
    bars2 = ax1.barh(y_pos, cap_factors, color=colors, edgecolor=COLORS['dark'], 
                     linewidth=1.0, alpha=0.8)
    
    bars2 = ax2.barh(y_pos, cap_factors, color=colors, edgecolor=COLORS['dark'], 
                     linewidth=1.0, alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_types, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Capacity Factor (%)', fontsize=16, fontweight='bold')
    ax2.set_title('Capacity Utilization by Fuel Type', fontsize=20, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 110)
    ax2.tick_params(axis='x', labelsize=13)
    
    # Add reference line at 100%
    ax2.axvline(x=100, color=COLORS['accent'], linestyle='--', linewidth=2.5, 
                label='Full Capacity')
    
    # Add value labels
    for bar, cf in zip(bars2, cap_factors):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{cf:.1f}%', va='center', ha='left', fontsize=12, fontweight='bold', color=COLORS['dark'])
    
    ax2.legend(loc='lower right', fontsize=13)
    
    # Add summary text
    fig.text(0.5, -0.02, f'Total Generation: {total_gen:,.0f} MW', 
             ha='center', fontsize=14, fontweight='bold', color=COLORS['primary'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Generation mix plot saved to {save_path}")
    
    return fig


# =============================================================================
# CONGESTION VISUALIZATION
# =============================================================================

def plot_congestion_enhanced(results: Dict, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create an enhanced congestion analysis visualization.
    
    Shows top congested lines with utilization levels and flow magnitudes.
    
    Parameters
    ----------
    results : dict
        Results dictionary containing 'congested_branches'
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    set_publication_style()
    
    if 'congested_branches' not in results or not results['congested_branches']:
        print("No congested branches to plot")
        return None
    
    congested = results['congested_branches']
    
    # Take top 15 most congested
    top_congested = sorted(congested, key=lambda x: abs(x['flow'])/x['rating'], reverse=True)[:15]
    
    branches = [cb['branch'] for cb in top_congested]
    flows = [abs(cb['flow']) for cb in top_congested]
    ratings = [cb['rating'] for cb in top_congested]
    utilizations = [f/r*100 for f, r in zip(flows, ratings)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    y_pos = np.arange(len(branches))
    
    # Color by utilization level
    colors = [COLORS['accent'] if u >= 99 else COLORS['warning'] if u >= 90 else COLORS['secondary'] 
              for u in utilizations]
    
    bars = ax.barh(y_pos, utilizations, color=colors, edgecolor=COLORS['dark'], linewidth=1.0)
    
    # Add capacity line
    ax.axvline(x=100, color=COLORS['accent'], linestyle='-', linewidth=3, 
               label='Thermal Limit (100%)')
    
    # Add warning threshold
    ax.axvline(x=90, color=COLORS['warning'], linestyle='--', linewidth=2, alpha=0.7,
               label='Warning Threshold (90%)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(branches, fontsize=12, fontweight='bold')
    ax.set_xlabel('Line Utilization (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Transmission Line', fontsize=16, fontweight='bold')
    ax.set_title('Transmission Line Congestion Analysis', fontsize=20, fontweight='bold')
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=13)
    
    # Add flow labels
    for bar, flow, rating in zip(bars, flows, ratings):
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{flow:.0f}/{rating:.0f} MW',
                va='center', ha='left', fontsize=11, fontweight='bold', color=COLORS['dark'])
    
    ax.legend(loc='lower right', framealpha=0.95, fontsize=13)
    
    # Add summary
    n_critical = sum(1 for u in utilizations if u >= 99)
    fig.text(0.5, -0.02, 
             f'{n_critical} lines at thermal limit | {len(utilizations)} lines shown',
             ha='center', fontsize=13, fontweight='bold', color=COLORS['primary'])
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Congestion plot saved to {save_path}")
    
    return fig


# =============================================================================
# COST SENSITIVITY VISUALIZATION
# =============================================================================

def plot_cost_sensitivity_enhanced(results_dir: Path, 
                                    output_path: Optional[str] = None) -> plt.Figure:
    """
    Create an enhanced cost sensitivity analysis visualization.
    
    Shows how transmission expansion decisions vary with capital costs.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing cost_sensitivity.csv
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    set_publication_style()
    
    csv_path = results_dir / "cost_sensitivity.csv"
    if not csv_path.exists():
        print(f"Cost sensitivity data not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path).sort_values("cost_per_mw")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    cost = df["cost_per_mw"] / 1000  # Convert to $/kMW
    op_cost = df["operating_cost"] / 1000  # Convert to $k
    
    # Main line: Operating cost
    line1, = ax1.semilogx(cost * 1000, op_cost, 
                          marker='o', markersize=12, linewidth=3.5,
                          color=COLORS['primary'], label='Operating Cost',
                          markeredgecolor=COLORS['dark'], markeredgewidth=1.5)
    
    ax1.set_xlabel('Line Capital Cost ($/MW, log scale)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Operating Cost ($k)', fontsize=16, fontweight='bold', color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'], labelsize=13)
    ax1.tick_params(axis='x', labelsize=13)
    
    # Baseline reference
    baseline = op_cost.iloc[0]
    ax1.axhline(baseline, color=COLORS['warning'], linestyle='--', linewidth=3,
                label=f'Baseline: ${baseline:,.0f}k')
    
    # Add annotation box
    ax1.annotate('No expansion triggered\nacross all cost levels',
                xy=(cost.iloc[len(cost)//2] * 1000, baseline),
                xytext=(cost.iloc[len(cost)//2] * 1000, baseline + 12),
                fontsize=13, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
                         edgecolor=COLORS['grid'], linewidth=2),
                arrowprops=dict(arrowstyle='->', color=COLORS['grid'], lw=2))
    
    # Secondary axis: Lines built
    ax2 = ax1.twinx()
    scatter = ax2.scatter(cost * 1000, df["lines_built"], 
                         color=COLORS['success'], marker='s', s=120, 
                         label='Lines Built', zorder=5,
                         edgecolors=COLORS['dark'], linewidths=1.5)
    ax2.set_ylabel('Lines Built', fontsize=16, fontweight='bold', color=COLORS['success'])
    ax2.tick_params(axis='y', labelcolor=COLORS['success'], labelsize=13)
    ax2.set_ylim(-0.5, max(df["lines_built"].max(), 1) + 0.5)
    
    ax1.set_title('Cost Sensitivity Analysis: Transmission Expansion Economics',
                  fontsize=20, fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95, fontsize=13)
    
    # Add grid
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Cost sensitivity plot saved to {output_path}")
    
    return fig


# =============================================================================
# LOAD SHEDDING VISUALIZATION
# =============================================================================

def plot_load_shedding_enhanced(results_dir: Path, 
                                 output_path: Optional[str] = None) -> plt.Figure:
    """
    Create an enhanced load shedding analysis visualization.
    
    Shows supply shortage stress test results with penalties.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing load_shedding_periods.csv
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    set_publication_style()
    
    csv_path = results_dir / "load_shedding_periods.csv"
    if not csv_path.exists():
        print(f"Load shedding data not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df["load_shed_cost_m"] = df["load_shed_cost"] / 1e6
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    periods = df["period"].astype(str)
    x = np.arange(len(periods))
    width = 0.6
    
    # Left plot: Load vs Generation with shedding
    ax1.bar(x - width/3, df["total_load_mw"], width/1.5, 
            label='Total Load', color=COLORS['accent'], 
            edgecolor=COLORS['dark'], linewidth=1.0, alpha=0.85)
    ax1.bar(x + width/3, df["total_generation_mw"], width/1.5,
            label='Generation', color=COLORS['success'],
            edgecolor=COLORS['dark'], linewidth=1.0, alpha=0.85)
    
    ax1.set_xlabel('Period', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Power (MW)', fontsize=16, fontweight='bold')
    ax1.set_title('Supply-Demand Gap Under Stress', fontsize=20, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelsize=13)
    ax1.legend(loc='upper left', fontsize=13)
    
    # Add load shed annotations
    max_load = df["total_load_mw"].max()
    n_periods = len(df)
    annot_fontsize = 10 if n_periods > 8 else 12
    for i, (load, gen, shed) in enumerate(zip(df["total_load_mw"], 
                                               df["total_generation_mw"],
                                               df["load_shed_mw"])):
        ax1.annotate(f'-{shed:.0f}',
                    xy=(i - width/3, load + max_load * 0.02),
                    ha='center', fontsize=annot_fontsize, color=COLORS['accent'],
                    fontweight='bold', rotation=45 if n_periods > 8 else 0)
    
    ax1.set_ylim(0, max_load * 1.18)
    
    # Right plot: Load shedding percentage and cost
    bars = ax2.bar(x, df["load_shed_pct"], width,
                   color=plt.cm.Reds(np.linspace(0.4, 0.8, len(df))),
                   edgecolor=COLORS['dark'], linewidth=1.0)
    
    ax2.set_xlabel('Period', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Load Shed (% of Demand)', fontsize=16, fontweight='bold', color=COLORS['accent'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'], labelsize=13)
    ax2.set_title('Load Shedding Magnitude & Penalty Cost', fontsize=20, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods, fontsize=13, fontweight='bold')
    
    # Add percentage labels
    for bar, pct in zip(bars, df["load_shed_pct"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f'{pct:.1f}%', ha='center', fontsize=12, fontweight='bold',
                color=COLORS['dark'])
    
    ax2.set_ylim(0, df["load_shed_pct"].max() * 1.25)
    
    # Secondary axis for cost
    ax3 = ax2.twinx()
    ax3.plot(x, df["load_shed_cost_m"], marker='D', markersize=12, 
             linewidth=3.5, color=COLORS['primary'],
             markeredgecolor=COLORS['dark'], markeredgewidth=1.5)
    ax3.set_ylabel('Penalty Cost ($M)', fontsize=16, fontweight='bold', color=COLORS['primary'])
    ax3.tick_params(axis='y', labelcolor=COLORS['primary'], labelsize=13)
    
    # Add title annotation
    fig.suptitle('Supply Shortage Stress Test: 20% Generator Availability, 40% Load Increase',
                 fontsize=16, fontweight='bold', y=0.98, color=COLORS['dark'])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Load shedding plot saved to {output_path}")
    
    return fig


# =============================================================================
# ROBUST TEP SCENARIOS VISUALIZATION
# =============================================================================

def plot_robust_scenarios_enhanced(results_dir: Path, 
                                    output_path: Optional[str] = None,
                                    data_loader=None) -> plt.Figure:
    """
    Create an enhanced robust TEP scenario comparison visualization.
    
    Shows scenario-level results and the network with new lines built.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing robust_tep_summary.csv
    output_path : str, optional
        Path to save the figure
    data_loader : RTSDataLoader, optional
        Data loader for network visualization
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    set_publication_style()
    
    csv_path = results_dir / "robust_tep_summary.csv"
    if not csv_path.exists():
        print(f"Robust TEP data not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    df["operating_cost_k"] = df["operating_cost"] / 1000
    
    # Create figure with 2 subplots: left = scenario comparison, right = network
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), 
                                    gridspec_kw={'width_ratios': [1, 1.3]})
    
    # =========================================================================
    # LEFT PLOT: Consolidated scenario comparison
    # =========================================================================
    scenarios = ['Base', 'High Load\nLow Renew', 'Low Load\nHigh Renew']
    x = np.arange(len(scenarios))
    width = 0.35
    
    # Normalize values for dual-axis display
    load_values = df["total_load_mw"].values
    cost_values = df["operating_cost_k"].values
    
    # Create bars for load (primary y-axis)
    bars1 = ax1.bar(x - width/2, load_values, width, 
                    label='System Load (MW)', color=COLORS['secondary'],
                    edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.9)
    
    ax1.set_ylabel('System Load (MW)', fontsize=16, fontweight='bold', color=COLORS['secondary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['secondary'], labelsize=13)
    ax1.set_ylim(0, max(load_values) * 1.25)
    
    # Add load value labels
    for bar, val in zip(bars1, load_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,.0f}\nMW', ha='center', fontsize=12, fontweight='bold',
                color=COLORS['secondary'])
    
    # Secondary axis for operating cost
    ax1_twin = ax1.twinx()
    bars2 = ax1_twin.bar(x + width/2, cost_values, width,
                         label='Operating Cost ($k)', color=COLORS['accent'],
                         edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.9)
    
    ax1_twin.set_ylabel('Operating Cost ($k)', fontsize=16, fontweight='bold', color=COLORS['accent'])
    ax1_twin.tick_params(axis='y', labelcolor=COLORS['accent'], labelsize=13)
    ax1_twin.set_ylim(0, max(cost_values) * 1.25)
    
    # Add cost value labels
    for bar, val in zip(bars2, cost_values):
        ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                     f'${val:,.0f}k', ha='center', fontsize=12, fontweight='bold',
                     color=COLORS['accent'])
    
    ax1.set_xlabel('Scenario', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, fontsize=13, fontweight='bold')
    ax1.set_title('Scenario Comparison: Load & Operating Cost', fontsize=18, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)
    
    # =========================================================================
    # RIGHT PLOT: Network with new lines highlighted
    # =========================================================================
    if data_loader is not None:
        buses = data_loader.get_bus_data()
        branches = data_loader.get_branch_data()
    else:
        # Try to load data if not provided
        try:
            import sys
            sys.path.insert(0, '.')
            from src.core.data_loader import RTSDataLoader
            data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
            buses = data_loader.get_bus_data()
            branches = data_loader.get_branch_data()
        except:
            ax2.text(0.5, 0.5, 'Network data not available', ha='center', va='center',
                    fontsize=16, transform=ax2.transAxes)
            ax2.axis('off')
            plt.tight_layout()
            if output_path:
                fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            return fig
    
    # Build network graph
    G = nx.Graph()
    pos = {}
    
    # Add nodes
    for bus_id in buses.index:
        G.add_node(bus_id)
        if 'lat' in buses.columns and 'lng' in buses.columns:
            pos[bus_id] = (buses.loc[bus_id, 'lng'], buses.loc[bus_id, 'lat'])
        else:
            pos[bus_id] = (bus_id % 10, bus_id // 10)
    
    # Add existing edges
    existing_edges = []
    for _, branch in branches.iterrows():
        from_bus = int(branch['From Bus'])
        to_bus = int(branch['To Bus'])
        existing_edges.append((from_bus, to_bus))
        G.add_edge(from_bus, to_bus)
    
    # Define new lines built by robust TEP
    new_lines = [(101, 106), (101, 117)]
    for from_bus, to_bus in new_lines:
        G.add_edge(from_bus, to_bus)
    
    # Color nodes by region
    region_colors_map = {1: '#3498DB', 2: '#E74C3C', 3: '#2ECC71'}
    node_colors = []
    node_sizes = []
    
    for bus_id in buses.index:
        region = buses.loc[bus_id, 'Area']
        # Highlight buses connected by new lines
        if bus_id in [101, 106, 117]:
            node_colors.append('#FFD700')  # Gold for key buses
            node_sizes.append(400)
        else:
            node_colors.append(region_colors_map.get(region, '#95A5A6'))
            node_sizes.append(80)
    
    # Draw existing edges (gray)
    nx.draw_networkx_edges(G, pos, edgelist=existing_edges,
                          edge_color=COLORS['grid'], width=1.0, alpha=0.5, ax=ax2)
    
    # Draw new lines (thick, highlighted)
    nx.draw_networkx_edges(G, pos, edgelist=new_lines,
                          edge_color='#E74C3C', width=5, alpha=1.0, ax=ax2,
                          style='solid')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                          alpha=0.9, edgecolors=COLORS['dark'], linewidths=1.0, ax=ax2)
    
    # Label key buses
    key_labels = {101: '101', 106: '106', 117: '117'}
    nx.draw_networkx_labels(G, pos, key_labels, font_size=12, font_weight='bold',
                           font_color='black', ax=ax2)
    
    # Add legend for network
    legend_elements = [
        mpatches.Patch(color='#3498DB', label='Region 1'),
        mpatches.Patch(color='#E74C3C', label='Region 2'),
        mpatches.Patch(color='#2ECC71', label='Region 3'),
        mpatches.Patch(color='#FFD700', label='Key Buses'),
        Line2D([0], [0], color='#E74C3C', linewidth=5, label='NEW LINES (200 MW each)'),
    ]
    legend = ax2.legend(handles=legend_elements, loc='upper left', fontsize=12,
                        title='Network Legend', title_fontsize=13)
    legend.get_title().set_fontweight('bold')
    
    ax2.set_title('Network with New Transmission Lines', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Longitude', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Latitude', fontsize=14, fontweight='bold')
    ax2.tick_params(labelsize=11)
    
    plt.suptitle('Scenario-Based Robust Transmission Expansion Planning',
                 fontsize=22, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Robust scenarios plot saved to {output_path}")
    
    return fig


# =============================================================================
# SUMMARY DASHBOARD
# =============================================================================

def create_summary_dashboard(results_dir: Path, data_loader,
                              baseline_results: Dict,
                              output_path: Optional[str] = None) -> plt.Figure:
    """Create a comprehensive summary dashboard with key metrics."""
    set_publication_style()
    
    fig = plt.figure(figsize=(20, 14))
    
    # Create grid layout: 3 rows, 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3, 
                          height_ratios=[0.8, 1.4, 1], width_ratios=[1, 1, 1, 1])
    
    # =========================================================================
    # TOP ROW: Key Metrics + Scenario Comparison
    # =========================================================================
    
    # 1. Key Metrics Panel (top left, spans 2 cols)
    ax_metrics = fig.add_subplot(gs[0, 0:2])
    ax_metrics.axis('off')
    
    metrics_text = """KEY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
System Buses: 73    |  Generators: 158  |  Branches: 120
Base Load: 8,550 MW |  Operating Cost: $138,967

Congested Lines: A27, CA-1, CB-1
Robust TEP Lines: Bus 101 → 106, Bus 101 → 117"""
    ax_metrics.text(0.02, 0.9, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=COLORS['light'], 
                            edgecolor=COLORS['grid'], linewidth=2))
    
    # 2. Scenario Comparison (top right, spans 2 cols)
    ax_scenario = fig.add_subplot(gs[0, 2:4])
    robust_file = results_dir / "robust_tep_summary.csv"
    if robust_file.exists():
        df = pd.read_csv(robust_file)
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax_scenario.bar(x - width/2, df["total_load_mw"], width, 
                               label='System Load (MW)', color=COLORS['secondary'], 
                               edgecolor=COLORS['dark'], linewidth=1.5)
        
        ax2 = ax_scenario.twinx()
        bars2 = ax2.bar(x + width/2, df["operating_cost"]/1000, width,
                       label='Operating Cost ($k)', color=COLORS['accent'], 
                       edgecolor=COLORS['dark'], linewidth=1.5)
        
        ax_scenario.set_xticks(x)
        scenario_labels = ['Base', 'High Load\nLow Renew', 'Low Load\nHigh Renew']
        ax_scenario.set_xticklabels(scenario_labels, fontsize=10, fontweight='bold')
        ax_scenario.set_ylabel('System Load (MW)', fontsize=11, fontweight='bold', color=COLORS['secondary'])
        ax2.set_ylabel('Operating Cost ($k)', fontsize=11, fontweight='bold', color=COLORS['accent'])
        ax_scenario.tick_params(axis='y', labelcolor=COLORS['secondary'], labelsize=10)
        ax2.tick_params(axis='y', labelcolor=COLORS['accent'], labelsize=10)
        
        # Add value labels
        for bar, val in zip(bars1, df["total_load_mw"]):
            ax_scenario.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                           f'{val:,.0f}\nMW', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, val in zip(bars2, df["operating_cost"]/1000):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                    f'${val:,.0f}k', ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['accent'])
        
        ax_scenario.legend(loc='upper left', fontsize=9)
        ax2.legend(loc='upper right', fontsize=9)
    ax_scenario.set_title('Scenario Comparison: Load & Operating Cost', fontsize=14, fontweight='bold')
    
    # =========================================================================
    # MIDDLE ROW: Large Network (spans all 4 columns)
    # =========================================================================
    
    ax_network = fig.add_subplot(gs[1, :])
    buses = data_loader.get_bus_data()
    branches = data_loader.get_branch_data()
    
    G = nx.Graph()
    pos = {}
    loads = {}
    for bus_id in buses.index:
        G.add_node(bus_id)
        if 'lat' in buses.columns:
            pos[bus_id] = (buses.loc[bus_id, 'lng'], buses.loc[bus_id, 'lat'])
        loads[bus_id] = buses.loc[bus_id, 'MW Load']
    
    for _, branch in branches.iterrows():
        G.add_edge(int(branch['From Bus']), int(branch['To Bus']), 
                  uid=branch['UID'], rating=branch['Cont Rating'])
    
    # Node sizes based on load (scaled larger)
    max_load = max(loads.values()) if loads.values() else 1
    node_sizes = [max(80, 400 * loads.get(b, 0) / max_load) for b in buses.index]
    
    region_colors_map = {1: '#3498DB', 2: '#E74C3C', 3: '#2ECC71'}
    node_colors = [region_colors_map.get(buses.loc[b, 'Area'], '#95A5A6') for b in buses.index]
    
    # Draw edges
    congested_uids = {'A27', 'CA-1', 'CB-1'}
    regular_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('uid') not in congested_uids]
    congested_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('uid') in congested_uids]
    
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges, ax=ax_network, 
                          edge_color=COLORS['grid'], alpha=0.4, width=1.2)
    nx.draw_networkx_edges(G, pos, edgelist=congested_edges, ax=ax_network, 
                          edge_color=COLORS['accent'], alpha=0.95, width=4)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax_network, node_color=node_colors, 
                          node_size=node_sizes, alpha=0.85, edgecolors='white', linewidths=1)
    
    # Draw new transmission lines (101→106, 101→117)
    new_lines = [(101, 106), (101, 117)]
    for from_bus, to_bus in new_lines:
        if from_bus in pos and to_bus in pos:
            ax_network.annotate('', xy=pos[to_bus], xytext=pos[from_bus],
                               arrowprops=dict(arrowstyle='-|>', color='#C0392B', lw=5, 
                                              mutation_scale=20))
    
    # Label key buses with larger circles
    key_buses = {101: '101', 106: '106', 117: '117'}
    for bus_id, label in key_buses.items():
        if bus_id in pos:
            ax_network.annotate(label, pos[bus_id], fontsize=11, fontweight='bold',
                               ha='center', va='center', color='white',
                               bbox=dict(boxstyle='circle', facecolor='#F39C12', 
                                        edgecolor='#E67E22', pad=0.4, linewidth=2))
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498DB', markersize=14, label='Region 1'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E74C3C', markersize=14, label='Region 2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ECC71', markersize=14, label='Region 3'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F39C12', markersize=14, label='Key Buses'),
        Line2D([0], [0], color=COLORS['accent'], linewidth=4, label='Congested Lines'),
        Line2D([0], [0], color='#C0392B', linewidth=5, label='NEW LINES (200 MW each)'),
    ]
    ax_network.legend(handles=legend_elements, loc='upper left', fontsize=11, 
                     framealpha=0.95, title='Network Legend', title_fontsize=12)
    ax_network.set_title('Network: Load Distribution, Congestion & Expansion', fontsize=16, fontweight='bold')
    ax_network.axis('off')
    
    # =========================================================================
    # BOTTOM ROW: Generation Mix + Load Shedding panels
    # =========================================================================
    
    # 3. Generation Mix - TREEMAP
    ax_gen = fig.add_subplot(gs[2, 0])
    if 'generation' in baseline_results:
        import squarify
        generators = data_loader.get_generator_data()
        gen_by_type = {}
        for gen_id, power in baseline_results['generation'].items():
            if power > 1:
                gen_row = generators[generators['GEN UID'] == gen_id]
                if len(gen_row) > 0:
                    gen_type = gen_row.iloc[0]['Unit Type']
                    gen_by_type[gen_type] = gen_by_type.get(gen_type, 0) + power
        
        if gen_by_type:
            total = sum(gen_by_type.values())
            large_types = {t: p for t, p in gen_by_type.items() if p/total >= 0.05}
            small_types = {t: p for t, p in gen_by_type.items() if p/total < 0.05}
            if small_types:
                large_types['Others'] = sum(small_types.values())
            
            sorted_items = sorted(large_types.items(), key=lambda x: x[1], reverse=True)
            types = [item[0] for item in sorted_items]
            powers = [item[1] for item in sorted_items]
            colors = [GEN_COLORS.get(t, '#7F8C8D') for t in types]
            labels = [f"{t}\n{p/total*100:.0f}%" for t, p in zip(types, powers)]
            
            squarify.plot(sizes=powers, label=labels, color=colors, alpha=0.85,
                         ax=ax_gen, text_kwargs={'fontsize': 9, 'fontweight': 'bold'})
            ax_gen.axis('off')
    ax_gen.set_title('Generation Mix', fontsize=14, fontweight='bold')
    
    # 4. Supply-Demand Gap (spans 2 cols)
    ax_gap = fig.add_subplot(gs[2, 1:3])
    shed_file = results_dir / "load_shedding_periods.csv"
    if shed_file.exists():
        df = pd.read_csv(shed_file)
        x = np.arange(len(df))
        width = 0.35
        ax_gap.bar(x - width/2, df["total_load_mw"], width, label='Total Load', 
                  color=COLORS['accent'], edgecolor=COLORS['dark'], alpha=0.8)
        ax_gap.bar(x + width/2, df["total_generation_mw"], width, label='Generation',
                  color=COLORS['success'], edgecolor=COLORS['dark'], alpha=0.8)
        ax_gap.set_xticks(x)
        ax_gap.set_xticklabels(df["period"], fontsize=10)
        ax_gap.set_xlabel('Period', fontsize=12, fontweight='bold')
        ax_gap.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
        ax_gap.legend(fontsize=10, loc='upper left')
        ax_gap.tick_params(labelsize=10)
    ax_gap.set_title('Supply-Demand Gap Under Stress', fontsize=14, fontweight='bold')
    
    # 5. Load Shedding & Cost
    ax_shed = fig.add_subplot(gs[2, 3])
    if shed_file.exists():
        df = pd.read_csv(shed_file)
        color_scale = plt.cm.Reds(np.linspace(0.3, 0.9, len(df)))
        ax_shed.bar(df["period"], df["load_shed_pct"], color=color_scale, 
                   edgecolor=COLORS['dark'], linewidth=1)
        ax_shed.set_xlabel('Period', fontsize=12, fontweight='bold')
        ax_shed.set_ylabel('Load Shed (%)', fontsize=12, fontweight='bold', color=COLORS['accent'])
        ax_shed.tick_params(labelsize=9)
        
        ax_cost = ax_shed.twinx()
        ax_cost.plot(df["period"], df["load_shed_cost"]/1e6, 'o-', color=COLORS['primary'], 
                    linewidth=2, markersize=4)
        ax_cost.set_ylabel('Penalty Cost ($M)', fontsize=11, fontweight='bold', color=COLORS['primary'])
        ax_cost.tick_params(axis='y', labelcolor=COLORS['primary'], labelsize=9)
    ax_shed.set_title('Load Shedding & Penalty Cost', fontsize=14, fontweight='bold')
    
    fig.suptitle('CME307 Transmission Expansion Planning - Results Summary',
                fontsize=20, fontweight='bold', y=0.98)
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Summary dashboard saved to {output_path}")
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def generate_all_visualizations(data_loader, baseline_results: Dict,
                                 results_dir: str = 'results') -> None:
    """
    Generate all enhanced visualizations for the project.
    
    Parameters
    ----------
    data_loader : RTSDataLoader
        Data loader instance
    baseline_results : dict
        Results from baseline DC-OPF
    results_dir : str
        Directory to save results
    """
    results_path = Path(results_dir)
    plots_path = results_path / 'plots'
    plots_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Generating Enhanced Visualizations")
    print("="*60 + "\n")
    
    # 1. Network topology
    print("1. Generating network topology plot...")
    plot_network_enhanced(
        data_loader, baseline_results,
        save_path=str(results_path / 'network_baseline.png'),
        title='RTS-GMLC Network: Baseline Congestion Analysis'
    )
    
    # 2. Generation mix
    print("\n2. Generating generation mix plot...")
    plot_generation_mix_enhanced(
        baseline_results, data_loader,
        save_path=str(results_path / 'generation_mix.png')
    )
    
    # 3. Congestion analysis
    print("\n3. Generating congestion analysis plot...")
    plot_congestion_enhanced(
        baseline_results,
        save_path=str(results_path / 'congestion_baseline.png')
    )
    
    # 4. Cost sensitivity
    print("\n4. Generating cost sensitivity plot...")
    plot_cost_sensitivity_enhanced(
        results_path,
        output_path=str(plots_path / 'plot_cost_sensitivity.png')
    )
    
    # 5. Load shedding
    print("\n5. Generating load shedding plot...")
    plot_load_shedding_enhanced(
        results_path,
        output_path=str(plots_path / 'plot_load_shedding.png')
    )
    
    # 6. Robust scenarios
    print("\n6. Generating robust scenarios plot...")
    plot_robust_scenarios_enhanced(
        results_path,
        output_path=str(plots_path / 'plot_robust_scenarios.png')
    )
    
    # 7. Summary dashboard
    print("\n7. Generating summary dashboard...")
    create_summary_dashboard(
        results_path, data_loader, baseline_results,
        output_path=str(plots_path / 'summary_dashboard.png')
    )
    
    print("\n" + "="*60)
    print("✓ All visualizations generated successfully!")
    print("="*60)


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.core.data_loader import RTSDataLoader
    from src.core.dc_opf import DCOPF
    
    print("="*60)
    print("CME307: Enhanced Visualization Generator")
    print("="*60)
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Run baseline
    print("Running baseline DC OPF...")
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    success = dcopf.solve(solver='gurobi')
    
    if success:
        baseline_results = dcopf.get_results()
        generate_all_visualizations(data_loader, baseline_results)
    else:
        print("Failed to solve baseline - cannot generate visualizations")
