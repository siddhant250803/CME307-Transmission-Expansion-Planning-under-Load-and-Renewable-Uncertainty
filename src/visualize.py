"""
Visualization tools for TEP results
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
from data_loader import RTSDataLoader

def plot_network(data_loader, results=None, save_path=None):
    """Plot the power network"""
    buses = data_loader.get_bus_data()
    branches = data_loader.get_branch_data()
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes with positions
    pos = {}
    for bus_id in buses.index:
        G.add_node(bus_id)
        if 'lat' in buses.columns and 'lng' in buses.columns:
            pos[bus_id] = (buses.loc[bus_id, 'lng'], buses.loc[bus_id, 'lat'])
        else:
            # Default positions
            pos[bus_id] = (bus_id % 10, bus_id // 10)
    
    # Add edges
    edge_colors = []
    edge_widths = []
    for _, branch in branches.iterrows():
        from_bus = int(branch['From Bus'])
        to_bus = int(branch['To Bus'])
        rating = branch['Cont Rating']
        
        G.add_edge(from_bus, to_bus, rating=rating)
        
        # Color by utilization if results available
        if results and 'flows' in results:
            flow = abs(results['flows'].get(branch['UID'], 0))
            utilization = flow / rating if rating > 0 else 0
            if utilization > 0.9:
                edge_colors.append('red')
                edge_widths.append(3.0)
            elif utilization > 0.7:
                edge_colors.append('orange')
                edge_widths.append(2.0)
            else:
                edge_colors.append('gray')
                edge_widths.append(1.0)
        else:
            edge_colors.append('gray')
            edge_widths.append(1.0)
    
    # Plot
    plt.figure(figsize=(16, 12))
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)
    
    # Add labels for important buses
    important_buses = [bus_id for bus_id in buses.index if bus_id % 100 in [13, 18, 21, 22, 23]]
    labels = {bus_id: str(bus_id) for bus_id in important_buses}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title('RTS-GMLC Network\nRed: >90% utilized, Orange: >70% utilized', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Network plot saved to {save_path}")
    else:
        plt.show()

def plot_congestion(results, save_path=None):
    """Plot congestion analysis"""
    if 'congested_branches' not in results or not results['congested_branches']:
        print("No congested branches to plot")
        return
    
    congested = results['congested_branches']
    branches = [cb['branch'] for cb in congested[:20]]  # Top 20
    utilizations = [abs(cb['flow']) / cb['rating'] * 100 for cb in congested[:20]]
    
    plt.figure(figsize=(12, 6))
    bars = plt.barh(branches, utilizations, color=['red' if u >= 99 else 'orange' for u in utilizations])
    plt.xlabel('Utilization (%)', fontsize=12)
    plt.title('Top Congested Transmission Lines', fontsize=14)
    plt.axvline(x=100, color='black', linestyle='--', linewidth=1)
    plt.xlim(0, 105)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Congestion plot saved to {save_path}")
    else:
        plt.show()

def plot_generation_mix(results, data_loader, save_path=None):
    """Plot generation mix"""
    if 'generation' not in results:
        print("No generation data to plot")
        return
    
    generators = data_loader.get_generator_data()
    gen_by_type = {}
    
    for gen_id, power in results['generation'].items():
        if power < 0.1:  # Skip very small generation
            continue
        gen_row = generators[generators['GEN UID'] == gen_id]
        if len(gen_row) > 0:
            gen_type = gen_row.iloc[0]['Unit Type']
            gen_by_type[gen_type] = gen_by_type.get(gen_type, 0) + power
    
    if not gen_by_type:
        print("No generation data available")
        return
    
    # Sort by generation
    types = list(gen_by_type.keys())
    powers = [gen_by_type[t] for t in types]
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Set3(range(len(types)))
    plt.pie(powers, labels=types, autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Generation Mix by Type', fontsize=14)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Generation mix plot saved to {save_path}")
    else:
        plt.show()

def create_results_report(baseline_results, tep_results, data_loader, output_dir='results'):
    """Create a comprehensive results report with visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations in '{output_dir}/' directory...")
    
    # Network plots
    plot_network(data_loader, baseline_results, 
                save_path=f'{output_dir}/network_baseline.png')
    
    if tep_results and 'lines_built' in tep_results and tep_results['lines_built']:
        plot_network(data_loader, tep_results, 
                    save_path=f'{output_dir}/network_tep.png')
    
    # Congestion plots
    plot_congestion(baseline_results, save_path=f'{output_dir}/congestion_baseline.png')
    
    # Generation mix
    plot_generation_mix(baseline_results, data_loader, 
                       save_path=f'{output_dir}/generation_mix.png')
    
    print(f"\nAll visualizations saved to '{output_dir}/' directory")


if __name__ == '__main__':
    import sys
    import os
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_loader import RTSDataLoader
    from src.dc_opf import DCOPF
    
    print("="*60)
    print("CME307: Transmission Expansion Planning Visualizations")
    print("="*60)
    
    # Load data
    print("\nLoading RTS-GMLC data...")
    data_loader = RTSDataLoader(data_dir='data/RTS_Data/SourceData')
    
    # Run baseline model
    print("Running baseline DC OPF...")
    dcopf = DCOPF(data_loader)
    dcopf.build_model()
    success = dcopf.solve(solver='gurobi')
    
    if not success:
        print("Failed to solve baseline model")
        sys.exit(1)
    
    baseline_results = dcopf.get_results()
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_results_report(baseline_results, None, data_loader, output_dir='results')
    
    print("\n" + "="*60)
    print("Visualizations complete!")
    print("="*60)

