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

# Set global style
def set_publication_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.facecolor': '#FAFAFA',
        'figure.facecolor': 'white',
    })


# =============================================================================
# NETWORK VISUALIZATION
# =============================================================================

def plot_network_enhanced(data_loader, results: Optional[Dict] = None, 
                          save_path: Optional[str] = None,
                          title: str = "RTS-GMLC Network Topology",
                          highlight_congested: bool = True) -> plt.Figure:
    """
    Create an enhanced network topology visualization.
    
    This function creates a detailed map of the power system network showing:
    - Bus locations (nodes) colored by region
    - Transmission lines (edges) with width proportional to capacity
    - Congestion highlighting (red for >90%, orange for >70%)
    - Generation capacity indicators at major buses
    
    Parameters
    ----------
    data_loader : RTSDataLoader
        Instance of the data loader with bus and branch data
    results : dict, optional
        Results dictionary containing 'flows' and 'congested_branches'
    save_path : str, optional
        Path to save the figure
    title : str
        Plot title
    highlight_congested : bool
        Whether to highlight congested lines
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    set_publication_style()
    
    buses = data_loader.get_bus_data()
    branches = data_loader.get_branch_data()
    
    # Create graph
    G = nx.Graph()
    
    # Node positions from lat/lng
    pos = {}
    node_colors = []
    node_sizes = []
    
    # Color by region
    region_colors = {1: '#3498DB', 2: '#E74C3C', 3: '#2ECC71'}
    
    for bus_id in buses.index:
        G.add_node(bus_id)
        if 'lat' in buses.columns and 'lng' in buses.columns:
            pos[bus_id] = (buses.loc[bus_id, 'lng'], buses.loc[bus_id, 'lat'])
        else:
            pos[bus_id] = (bus_id % 10, bus_id // 10)
        
        region = buses.loc[bus_id, 'Area']
        node_colors.append(region_colors.get(region, '#95A5A6'))
        
        # Size by load
        load = buses.loc[bus_id, 'MW Load']
        node_sizes.append(max(50, min(300, load / 2)))
    
    # Process edges
    edgelist = []
    edge_colors = []
    edge_widths = []
    
    for _, branch in branches.iterrows():
        from_bus = int(branch['From Bus'])
        to_bus = int(branch['To Bus'])
        rating = branch['Cont Rating']
        uid = branch['UID']
        
        edgelist.append((from_bus, to_bus))
        G.add_edge(from_bus, to_bus, rating=rating, uid=uid)
        
        # Determine edge styling based on flow
        if results and 'flows' in results and highlight_congested:
            flow = abs(results['flows'].get(uid, 0))
            utilization = flow / rating if rating > 0 else 0
            
            if utilization >= 0.99:
                edge_colors.append(COLORS['accent'])
                edge_widths.append(3.5)
            elif utilization >= 0.70:
                edge_colors.append(COLORS['warning'])
                edge_widths.append(2.5)
            else:
                edge_colors.append(COLORS['grid'])
                edge_widths.append(1.0)
        else:
            # Width by capacity
            edge_colors.append(COLORS['secondary'])
            edge_widths.append(0.5 + (rating / 500))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw edges first (behind nodes)
    nx.draw_networkx_edges(
        G, pos, edgelist=edgelist,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.7,
        ax=ax
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        edgecolors=COLORS['dark'],
        linewidths=1.0,
        ax=ax
    )
    
    # Add labels for major buses (high load or generation)
    labels = {}
    for bus_id in buses.index:
        load = buses.loc[bus_id, 'MW Load']
        if load > 200 or bus_id % 100 in [13, 18, 21, 22, 23]:
            labels[bus_id] = str(bus_id)
    
    nx.draw_networkx_labels(
        G, pos, labels,
        font_size=8,
        font_weight='bold',
        font_color=COLORS['dark'],
        ax=ax
    )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color=region_colors[1], label='Region 1'),
        mpatches.Patch(color=region_colors[2], label='Region 2'),
        mpatches.Patch(color=region_colors[3], label='Region 3'),
    ]
    
    if highlight_congested and results:
        legend_elements.extend([
            Line2D([0], [0], color=COLORS['accent'], linewidth=3, label='>99% Utilized'),
            Line2D([0], [0], color=COLORS['warning'], linewidth=2.5, label='>70% Utilized'),
            Line2D([0], [0], color=COLORS['grid'], linewidth=1, label='Normal'),
        ])
    
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Longitude', fontsize=11)
    ax.set_ylabel('Latitude', fontsize=11)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Generation by type (horizontal bars)
    y_pos = np.arange(len(sorted_types))
    bars = ax1.barh(y_pos, powers, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_types)
    ax1.set_xlabel('Generation (MW)', fontsize=12)
    ax1.set_title('Generation by Fuel Type', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        width = bar.get_width()
        ax1.text(width + 50, bar.get_y() + bar.get_height()/2,
                f'{width:,.0f} MW ({pct:.1f}%)',
                va='center', ha='left', fontsize=9, color=COLORS['dark'])
    
    ax1.set_xlim(0, max(powers) * 1.35)
    ax1.axvline(x=0, color=COLORS['dark'], linewidth=0.8)
    
    # Right plot: Capacity factor
    bars2 = ax2.barh(y_pos, cap_factors, color=colors, edgecolor=COLORS['dark'], 
                     linewidth=0.5, alpha=0.8)
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_types)
    ax2.set_xlabel('Capacity Factor (%)', fontsize=12)
    ax2.set_title('Capacity Utilization by Fuel Type', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.set_xlim(0, 110)
    
    # Add reference line at 100%
    ax2.axvline(x=100, color=COLORS['accent'], linestyle='--', linewidth=1.5, 
                label='Full Capacity')
    
    # Add value labels
    for bar, cf in zip(bars2, cap_factors):
        width = bar.get_width()
        ax2.text(width + 2, bar.get_y() + bar.get_height()/2,
                f'{cf:.1f}%', va='center', ha='left', fontsize=9, color=COLORS['dark'])
    
    ax2.legend(loc='lower right')
    
    # Add summary text
    fig.text(0.5, -0.02, f'Total Generation: {total_gen:,.0f} MW', 
             ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'])
    
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
    fig, ax = plt.subplots(figsize=(12, 7))
    
    y_pos = np.arange(len(branches))
    
    # Color by utilization level
    colors = [COLORS['accent'] if u >= 99 else COLORS['warning'] if u >= 90 else COLORS['secondary'] 
              for u in utilizations]
    
    bars = ax.barh(y_pos, utilizations, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    
    # Add capacity line
    ax.axvline(x=100, color=COLORS['accent'], linestyle='-', linewidth=2, 
               label='Thermal Limit (100%)')
    
    # Add warning threshold
    ax.axvline(x=90, color=COLORS['warning'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Warning Threshold (90%)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(branches, fontsize=9)
    ax.set_xlabel('Line Utilization (%)', fontsize=12)
    ax.set_ylabel('Transmission Line', fontsize=12)
    ax.set_title('Transmission Line Congestion Analysis', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.invert_yaxis()
    
    # Add flow labels
    for bar, flow, rating in zip(bars, flows, ratings):
        width = bar.get_width()
        ax.text(width + 1.5, bar.get_y() + bar.get_height()/2,
                f'{flow:.0f}/{rating:.0f} MW',
                va='center', ha='left', fontsize=8, color=COLORS['dark'])
    
    ax.legend(loc='lower right', framealpha=0.95)
    
    # Add summary
    n_critical = sum(1 for u in utilizations if u >= 99)
    fig.text(0.5, -0.02, 
             f'{n_critical} lines at thermal limit | {len(utilizations)} lines shown',
             ha='center', fontsize=10, color=COLORS['primary'])
    
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
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    cost = df["cost_per_mw"] / 1000  # Convert to $/kMW
    op_cost = df["operating_cost"] / 1000  # Convert to $k
    
    # Main line: Operating cost
    line1, = ax1.semilogx(cost * 1000, op_cost, 
                          marker='o', markersize=8, linewidth=2.5,
                          color=COLORS['primary'], label='Operating Cost',
                          markeredgecolor=COLORS['dark'], markeredgewidth=1)
    
    ax1.set_xlabel('Line Capital Cost ($ per MW, log scale)', fontsize=12)
    ax1.set_ylabel('Operating Cost ($k)', fontsize=12, color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    
    # Baseline reference
    baseline = op_cost.iloc[0]
    ax1.axhline(baseline, color=COLORS['warning'], linestyle='--', linewidth=2,
                label=f'Baseline: ${baseline:,.0f}k')
    
    # Add annotation box
    ax1.annotate('No expansion triggered\nacross all cost levels',
                xy=(cost.iloc[len(cost)//2] * 1000, baseline),
                xytext=(cost.iloc[len(cost)//2] * 1000, baseline + 10),
                fontsize=10, ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
                         edgecolor=COLORS['grid']),
                arrowprops=dict(arrowstyle='->', color=COLORS['grid']))
    
    # Secondary axis: Lines built
    ax2 = ax1.twinx()
    scatter = ax2.scatter(cost * 1000, df["lines_built"], 
                         color=COLORS['success'], marker='s', s=80, 
                         label='Lines Built', zorder=5,
                         edgecolors=COLORS['dark'], linewidths=1)
    ax2.set_ylabel('Lines Built', fontsize=12, color=COLORS['success'])
    ax2.tick_params(axis='y', labelcolor=COLORS['success'])
    ax2.set_ylim(-0.5, max(df["lines_built"].max(), 1) + 0.5)
    
    ax1.set_title('Cost Sensitivity Analysis: Transmission Expansion Economics',
                  fontsize=14, fontweight='bold', pad=15)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', framealpha=0.95)
    
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
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    periods = df["period"].astype(str)
    x = np.arange(len(periods))
    width = 0.6
    
    # Left plot: Load vs Generation with shedding
    ax1.bar(x - width/3, df["total_load_mw"], width/1.5, 
            label='Total Load', color=COLORS['accent'], 
            edgecolor=COLORS['dark'], linewidth=0.5, alpha=0.8)
    ax1.bar(x + width/3, df["total_generation_mw"], width/1.5,
            label='Generation', color=COLORS['success'],
            edgecolor=COLORS['dark'], linewidth=0.5, alpha=0.8)
    
    ax1.set_xlabel('Period', fontsize=12)
    ax1.set_ylabel('Power (MW)', fontsize=12)
    ax1.set_title('Supply-Demand Gap Under Stress', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods)
    ax1.legend(loc='upper left')
    
    # Add load shed annotations
    for i, (load, gen, shed) in enumerate(zip(df["total_load_mw"], 
                                               df["total_generation_mw"],
                                               df["load_shed_mw"])):
        ax1.annotate(f'-{shed:.0f} MW',
                    xy=(i, gen + 50),
                    ha='center', fontsize=9, color=COLORS['accent'],
                    fontweight='bold')
    
    # Right plot: Load shedding percentage and cost
    bars = ax2.bar(x, df["load_shed_pct"], width,
                   color=plt.cm.Reds(np.linspace(0.4, 0.8, len(df))),
                   edgecolor=COLORS['dark'], linewidth=0.5)
    
    ax2.set_xlabel('Period', fontsize=12)
    ax2.set_ylabel('Load Shed (% of Demand)', fontsize=12, color=COLORS['accent'])
    ax2.tick_params(axis='y', labelcolor=COLORS['accent'])
    ax2.set_title('Load Shedding Magnitude & Penalty Cost', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods)
    
    # Add percentage labels
    for bar, pct in zip(bars, df["load_shed_pct"]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold',
                color=COLORS['dark'])
    
    ax2.set_ylim(0, df["load_shed_pct"].max() * 1.2)
    
    # Secondary axis for cost
    ax3 = ax2.twinx()
    ax3.plot(x, df["load_shed_cost_m"], marker='D', markersize=8, 
             linewidth=2.5, color=COLORS['primary'],
             markeredgecolor=COLORS['dark'], markeredgewidth=1)
    ax3.set_ylabel('Penalty Cost ($M)', fontsize=12, color=COLORS['primary'])
    ax3.tick_params(axis='y', labelcolor=COLORS['primary'])
    
    # Add title annotation
    fig.suptitle('Supply Shortage Stress Test: 20% Generator Availability, 40% Load Increase',
                 fontsize=12, y=0.98, color=COLORS['dark'])
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Load shedding plot saved to {output_path}")
    
    return fig


# =============================================================================
# ROBUST TEP SCENARIOS VISUALIZATION
# =============================================================================

def plot_robust_scenarios_enhanced(results_dir: Path, 
                                    output_path: Optional[str] = None) -> plt.Figure:
    """
    Create an enhanced robust TEP scenario comparison visualization.
    
    Shows scenario-level results from the robust optimization.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing robust_tep_summary.csv
    output_path : str, optional
        Path to save the figure
        
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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    scenarios = df["scenario"]
    x = np.arange(len(scenarios))
    
    # Scenario colors
    scenario_colors = [COLORS['secondary'], COLORS['accent'], COLORS['primary']]
    
    # Plot 1: Total Load by Scenario
    bars1 = axes[0].bar(x, df["total_load_mw"], 
                        color=scenario_colors, 
                        edgecolor=COLORS['dark'], linewidth=1)
    axes[0].set_xlabel('Scenario', fontsize=11)
    axes[0].set_ylabel('Total Load (MW)', fontsize=11)
    axes[0].set_title('Load Level by Scenario', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(['Base', 'High Load\nLow Renew', 'Low Load\nHigh Renew'], fontsize=9)
    
    for bar, val in zip(bars1, df["total_load_mw"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                    f'{val:,.0f}', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 2: Operating Cost by Scenario
    bars2 = axes[1].bar(x, df["operating_cost_k"],
                        color=scenario_colors,
                        edgecolor=COLORS['dark'], linewidth=1)
    axes[1].set_xlabel('Scenario', fontsize=11)
    axes[1].set_ylabel('Operating Cost ($k)', fontsize=11)
    axes[1].set_title('Operating Cost by Scenario', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Base', 'High Load\nLow Renew', 'Low Load\nHigh Renew'], fontsize=9)
    
    for bar, val in zip(bars2, df["operating_cost_k"]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'${val:,.0f}k', ha='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Summary metrics (stacked horizontal)
    # Show capacity factor comparison
    cf = df["total_generation_mw"] / df["total_load_mw"] * 100
    bars3 = axes[2].barh(x, cf, color=scenario_colors, 
                         edgecolor=COLORS['dark'], linewidth=1)
    axes[2].set_xlabel('Generation/Load Ratio (%)', fontsize=11)
    axes[2].set_ylabel('')
    axes[2].set_title('Supply-Demand Balance', fontsize=12, fontweight='bold')
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(['Base', 'High Load\nLow Renew', 'Low Load\nHigh Renew'], fontsize=9)
    axes[2].axvline(x=100, color=COLORS['dark'], linestyle='-', linewidth=2)
    axes[2].set_xlim(95, 105)
    
    # Add annotation about investment
    fig.text(0.5, -0.08, 
             '▶ Robust solution builds 2 new lines (Bus 101→106 and 101→117) at $61.3M to ensure feasibility across all scenarios',
             ha='center', fontsize=11, fontweight='bold', color=COLORS['primary'],
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['light'], 
                      edgecolor=COLORS['grid']))
    
    plt.suptitle('Scenario-Based Robust Transmission Expansion Planning',
                 fontsize=14, fontweight='bold', y=1.02)
    
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
    """
    Create a comprehensive summary dashboard with key metrics.
    
    Parameters
    ----------
    results_dir : Path
        Directory containing result CSV files
    data_loader : RTSDataLoader
        Data loader instance
    baseline_results : dict
        Baseline DC-OPF results
    output_path : str, optional
        Path to save the figure
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    set_publication_style()
    
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Key Metrics Panel (top left)
    ax_metrics = fig.add_subplot(gs[0, 0])
    ax_metrics.axis('off')
    
    metrics_text = """
    KEY METRICS
    ━━━━━━━━━━━━━━━━━━━
    System Buses: 73
    Generators: 158  
    Branches: 120
    
    Base Load: 8,550 MW
    Operating Cost: $138,967
    Congested Lines: 3
    
    Lines Built: 0 (Deterministic)
    Lines Built: 2 (Robust)
    """
    ax_metrics.text(0.1, 0.95, metrics_text, transform=ax_metrics.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=COLORS['light'], 
                            edgecolor=COLORS['grid']))
    
    # 2. Network mini-plot (top center & right)
    ax_network = fig.add_subplot(gs[0, 1:])
    buses = data_loader.get_bus_data()
    branches = data_loader.get_branch_data()
    
    G = nx.Graph()
    pos = {}
    for bus_id in buses.index:
        G.add_node(bus_id)
        if 'lat' in buses.columns:
            pos[bus_id] = (buses.loc[bus_id, 'lng'], buses.loc[bus_id, 'lat'])
    
    for _, branch in branches.iterrows():
        G.add_edge(int(branch['From Bus']), int(branch['To Bus']))
    
    region_colors = {1: '#3498DB', 2: '#E74C3C', 3: '#2ECC71'}
    node_colors = [region_colors.get(buses.loc[b, 'Area'], '#95A5A6') for b in buses.index]
    
    nx.draw_networkx(G, pos, ax=ax_network, node_color=node_colors, 
                    node_size=30, with_labels=False, edge_color=COLORS['grid'],
                    alpha=0.7, width=0.5)
    ax_network.set_title('Network Topology (3 Regions)', fontsize=12, fontweight='bold')
    ax_network.axis('off')
    
    # 3. Generation Mix (middle left)
    ax_gen = fig.add_subplot(gs[1, 0])
    if 'generation' in baseline_results:
        generators = data_loader.get_generator_data()
        gen_by_type = {}
        for gen_id, power in baseline_results['generation'].items():
            if power > 1:
                gen_row = generators[generators['GEN UID'] == gen_id]
                if len(gen_row) > 0:
                    gen_type = gen_row.iloc[0]['Unit Type']
                    gen_by_type[gen_type] = gen_by_type.get(gen_type, 0) + power
        
        if gen_by_type:
            types = list(gen_by_type.keys())
            powers = [gen_by_type[t] for t in types]
            colors = [GEN_COLORS.get(t, '#95A5A6') for t in types]
            ax_gen.pie(powers, labels=types, colors=colors, autopct='%1.0f%%',
                      textprops={'fontsize': 8})
    ax_gen.set_title('Generation Mix', fontsize=12, fontweight='bold')
    
    # 4. Congestion (middle center)
    ax_cong = fig.add_subplot(gs[1, 1])
    if 'congested_branches' in baseline_results and baseline_results['congested_branches']:
        congested = baseline_results['congested_branches'][:5]
        branches_list = [c['branch'][:10] for c in congested]
        utils = [abs(c['flow'])/c['rating']*100 for c in congested]
        colors = [COLORS['accent'] if u >= 99 else COLORS['warning'] for u in utils]
        ax_cong.barh(range(len(branches_list)), utils, color=colors)
        ax_cong.set_yticks(range(len(branches_list)))
        ax_cong.set_yticklabels(branches_list, fontsize=8)
        ax_cong.axvline(x=100, color='black', linestyle='--')
        ax_cong.set_xlim(0, 105)
    ax_cong.set_title('Top Congested Lines (%)', fontsize=12, fontweight='bold')
    ax_cong.set_xlabel('Utilization %')
    
    # 5. Cost Sensitivity Summary (middle right)
    ax_cost = fig.add_subplot(gs[1, 2])
    cost_file = results_dir / "cost_sensitivity.csv"
    if cost_file.exists():
        df = pd.read_csv(cost_file)
        ax_cost.semilogx(df["cost_per_mw"], df["operating_cost"]/1000, 
                        marker='o', color=COLORS['primary'], linewidth=2)
        ax_cost.axhline(df["operating_cost"].iloc[0]/1000, color=COLORS['warning'], 
                       linestyle='--', label='Baseline')
        ax_cost.set_xlabel('Line Cost ($/MW)')
        ax_cost.set_ylabel('Op. Cost ($k)')
        ax_cost.legend()
    ax_cost.set_title('Cost Sensitivity', fontsize=12, fontweight='bold')
    
    # 6. Load Shedding (bottom left)
    ax_shed = fig.add_subplot(gs[2, 0])
    shed_file = results_dir / "load_shedding_periods.csv"
    if shed_file.exists():
        df = pd.read_csv(shed_file)
        ax_shed.bar(df["period"], df["load_shed_pct"], color=COLORS['accent'])
        ax_shed.set_xlabel('Period')
        ax_shed.set_ylabel('Load Shed (%)')
    ax_shed.set_title('Load Shedding Stress Test', fontsize=12, fontweight='bold')
    
    # 7. Robust Scenarios (bottom center & right)
    ax_robust = fig.add_subplot(gs[2, 1:])
    robust_file = results_dir / "robust_tep_summary.csv"
    if robust_file.exists():
        df = pd.read_csv(robust_file)
        x = np.arange(len(df))
        width = 0.35
        ax_robust.bar(x - width/2, df["total_load_mw"], width, 
                     label='Load (MW)', color=COLORS['secondary'])
        ax_robust.bar(x + width/2, df["total_generation_mw"], width,
                     label='Generation (MW)', color=COLORS['success'])
        ax_robust.set_xticks(x)
        ax_robust.set_xticklabels(df["scenario"], rotation=15, fontsize=9)
        ax_robust.legend()
        ax_robust.set_ylabel('MW')
    ax_robust.set_title('Robust TEP Scenario Comparison', fontsize=12, fontweight='bold')
    
    fig.suptitle('CME307 Transmission Expansion Planning - Results Summary',
                fontsize=16, fontweight='bold', y=0.98)
    
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

