#!/usr/bin/env python3
"""
InvBankAI System Architecture Diagram Generator
Creates a visual representation of the data flow and component dependencies.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_diagram():
    # Create figure with larger size for detailed diagram
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Define colors for different component types
    colors = {
        'data_source': '#E3F2FD',      # Light blue - data sources
        'processing': '#F3E5F5',       # Light purple - processing
        'storage': '#E8F5E8',          # Light green - storage
        'model': '#FFF3E0',            # Light orange - ML components
        'trading': '#FFEBEE',          # Light red - trading
        'config': '#F5F5F5'            # Light gray - configuration
    }
    
    # Define component positions and information
    components = {
        # Data Sources (Top Row)
        'polygon': {'pos': (1, 12), 'size': (2.5, 1), 'color': colors['data_source'], 'text': 'Polygon.io\nFinancial News', 'type': 'source'},
        'newsapi': {'pos': (4, 12), 'size': (2.5, 1), 'color': colors['data_source'], 'text': 'NewsAPI\nGeneral News', 'type': 'source'},
        'alphavantage': {'pos': (7, 12), 'size': (2.5, 1), 'color': colors['data_source'], 'text': 'Alpha Vantage\nFinancial Data', 'type': 'source'},
        'fmp': {'pos': (10, 12), 'size': (2.5, 1), 'color': colors['data_source'], 'text': 'FMP\nCompany News', 'type': 'source'},
        'twitter': {'pos': (13, 12), 'size': (2.5, 1), 'color': colors['data_source'], 'text': 'Twitter API\nSocial Sentiment', 'type': 'source'},
        'reddit': {'pos': (16, 12), 'size': (2.5, 1), 'color': colors['data_source'], 'text': 'Reddit API\nFinancial Subs', 'type': 'source'},
        'yfinance': {'pos': (1, 10), 'size': (2.5, 1), 'color': colors['data_source'], 'text': 'yfinance\nPrice Data', 'type': 'source'},
        
        # Processing Layer (Middle-Upper)
        'fh_getsent': {'pos': (4, 9.5), 'size': (12, 1.5), 'color': colors['processing'], 'text': 'FH_getSent.py\nEnhanced Multi-Source Sentiment Analysis\n• FinBERT (40%) + RoBERTa (35%) + DistilBERT (25%)\n• Weighted scoring, engagement metrics\n• Momentum & velocity calculations', 'type': 'processing'},
        'download_data': {'pos': (1, 7.5), 'size': (3, 1), 'color': colors['processing'], 'text': 'dowload_data.py\nPrice Data Fetcher', 'type': 'processing'},
        'compute_ema': {'pos': (5, 7.5), 'size': (3, 1), 'color': colors['processing'], 'text': 'compute_ema.py\nTechnical Indicators\n(EMA, RSI, MACD)', 'type': 'processing'},
        
        # Data Storage (Middle)
        'raw_data': {'pos': (1, 5.5), 'size': (3, 1), 'color': colors['storage'], 'text': 'data/raw/\n{TICKER}_raw.csv', 'type': 'storage'},
        'sentiment_data': {'pos': (5, 5.5), 'size': (4, 1), 'color': colors['storage'], 'text': 'data/news/\n{TICKER}_sentiment_combined.csv', 'type': 'storage'},
        'features_data': {'pos': (10, 5.5), 'size': (4, 1), 'color': colors['storage'], 'text': 'data/features/\n{TICKER}_features.csv', 'type': 'storage'},
        
        # Data Processing Pipeline
        'data_manager': {'pos': (1, 3.5), 'size': (6, 1), 'color': colors['processing'], 'text': 'Data_manager.ipynb\nOrchestrates Pipeline Execution', 'type': 'processing'},
        'feel_manager': {'pos': (8, 3.5), 'size': (6, 1), 'color': colors['processing'], 'text': 'FeelManager.ipynb\nCombines Data + Technical Analysis', 'type': 'processing'},
        
        # Final Dataset
        'final_data': {'pos': (15, 5.5), 'size': (4, 1), 'color': colors['storage'], 'text': 'data/final/\n{TICKER}_ready.csv', 'type': 'storage'},
        
        # ML Components
        'trading_env': {'pos': (1, 1.5), 'size': (4, 1), 'color': colors['model'], 'text': 'TradingEnv\nGym Environment\n(buy/hold/sell actions)', 'type': 'model'},
        'ppo_training': {'pos': (6, 1.5), 'size': (4, 1), 'color': colors['model'], 'text': 'train_ppo_agent_logged.py\nPPO Model Training', 'type': 'model'},
        'trained_model': {'pos': (11, 1.5), 'size': (3, 1), 'color': colors['storage'], 'text': 'model/\nppo_{ticker}_agent.zip', 'type': 'storage'},
        
        # Trading Components
        'trade_agent': {'pos': (15, 1.5), 'size': (4, 1), 'color': colors['trading'], 'text': 'trade_agent.py\nLive Trading Agent', 'type': 'trading'},
        'alpaca_api': {'pos': (16.5, 0), 'size': (2.5, 1), 'color': colors['trading'], 'text': 'Alpaca API\nPaper Trading', 'type': 'trading'},
        
        # Configuration
        'config': {'pos': (0.5, 0), 'size': (3, 1), 'color': colors['config'], 'text': 'config.py\nTicker Configuration\nPath Management', 'type': 'config'},
    }
    
    # Draw components
    for name, comp in components.items():
        x, y = comp['pos']
        w, h = comp['size']
        
        # Create rounded rectangle
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=comp['color'],
            edgecolor='black',
            linewidth=1.5 if comp['type'] == 'processing' else 1
        )
        ax.add_patch(box)
        
        # Add text
        ax.text(x + w/2, y + h/2, comp['text'], 
                ha='center', va='center', fontsize=8,
                weight='bold' if comp['type'] == 'processing' else 'normal',
                wrap=True)
    
    # Define data flow connections
    connections = [
        # Data sources to sentiment processing
        ('polygon', 'fh_getsent'),
        ('newsapi', 'fh_getsent'),
        ('alphavantage', 'fh_getsent'),
        ('fmp', 'fh_getsent'),
        ('twitter', 'fh_getsent'),
        ('reddit', 'fh_getsent'),
        
        # Price data flow
        ('yfinance', 'download_data'),
        ('download_data', 'raw_data'),
        ('raw_data', 'compute_ema'),
        ('compute_ema', 'features_data'),
        
        # Sentiment data flow
        ('fh_getsent', 'sentiment_data'),
        
        # Pipeline orchestration
        ('data_manager', 'feel_manager'),
        ('sentiment_data', 'feel_manager'),
        ('features_data', 'feel_manager'),
        ('feel_manager', 'final_data'),
        
        # ML pipeline
        ('final_data', 'trading_env'),
        ('trading_env', 'ppo_training'),
        ('ppo_training', 'trained_model'),
        
        # Trading pipeline
        ('trained_model', 'trade_agent'),
        ('trade_agent', 'alpaca_api'),
        
        # Configuration
        ('config', 'download_data'),
        ('config', 'compute_ema'),
        ('config', 'fh_getsent'),
    ]
    
    # Draw connections
    for start, end in connections:
        start_comp = components[start]
        end_comp = components[end]
        
        # Calculate connection points
        start_x = start_comp['pos'][0] + start_comp['size'][0] / 2
        start_y = start_comp['pos'][1]
        end_x = end_comp['pos'][0] + end_comp['size'][0] / 2
        end_y = end_comp['pos'][1] + end_comp['size'][1]
        
        # Adjust for better visual flow
        if start_y > end_y:  # Downward flow
            start_y = start_comp['pos'][1]
            end_y = end_comp['pos'][1] + end_comp['size'][1]
        else:  # Upward or horizontal flow
            start_y = start_comp['pos'][1] + start_comp['size'][1]
            end_y = end_comp['pos'][1]
        
        # Draw arrow
        arrow = ConnectionPatch(
            (start_x, start_y), (end_x, end_y),
            "data", "data",
            arrowstyle="->",
            shrinkA=5, shrinkB=5,
            mutation_scale=20,
            fc="darkblue",
            alpha=0.6,
            linewidth=1.5
        )
        ax.add_patch(arrow)
    
    # Add title and sections
    ax.text(10, 13.5, 'InvBankAI System Architecture', ha='center', va='center', 
            fontsize=20, weight='bold')
    
    # Add section labels
    section_labels = [
        ('Data Sources', 9.5, 12.5, 'darkblue'),
        ('Sentiment & Price Processing', 9.5, 9, 'purple'),
        ('Data Storage', 9.5, 6, 'darkgreen'),
        ('Data Pipeline', 7, 4, 'purple'),
        ('ML Training & Trading', 9.5, 2, 'darkorange'),
    ]
    
    for label, x, y, color in section_labels:
        ax.text(x, y, label, ha='center', va='center', 
                fontsize=12, weight='bold', color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Add legend
    legend_elements = [
        ('Data Sources', colors['data_source']),
        ('Processing', colors['processing']),
        ('Storage', colors['storage']),
        ('ML Components', colors['model']),
        ('Trading', colors['trading']),
        ('Configuration', colors['config'])
    ]
    
    legend_y = 11
    for i, (label, color) in enumerate(legend_elements):
        y_pos = legend_y - i * 0.4
        rect = patches.Rectangle((0.2, y_pos), 0.3, 0.3, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.6, y_pos + 0.15, label, va='center', fontsize=9)
    
    ax.text(0.35, legend_y + 0.8, 'Component Types', ha='center', va='center', 
            fontsize=10, weight='bold')
    
    plt.tight_layout()
    return fig

def create_dependency_matrix():
    """Create a dependency matrix showing component relationships"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Define components in execution order
    components = [
        'config.py', 'FH_getSent.py', 'dowload_data.py', 'compute_ema.py',
        'Data_manager.ipynb', 'FeelManager.ipynb', 'TradingEnv',
        'train_ppo_agent.py', 'trade_agent.py', 'Alpaca API'
    ]
    
    # Define dependencies (row depends on column)
    dependencies = {
        'config.py': [],
        'FH_getSent.py': ['config.py'],
        'dowload_data.py': ['config.py'],
        'compute_ema.py': ['config.py', 'dowload_data.py'],
        'Data_manager.ipynb': ['FH_getSent.py', 'dowload_data.py', 'compute_ema.py'],
        'FeelManager.ipynb': ['Data_manager.ipynb'],
        'TradingEnv': ['FeelManager.ipynb'],
        'train_ppo_agent.py': ['TradingEnv', 'config.py'],
        'trade_agent.py': ['train_ppo_agent.py', 'config.py'],
        'Alpaca API': ['trade_agent.py']
    }
    
    # Create dependency matrix
    n = len(components)
    matrix = np.zeros((n, n))
    
    for i, comp in enumerate(components):
        for dep in dependencies[comp]:
            j = components.index(dep)
            matrix[i, j] = 1
    
    # Plot matrix
    im = ax.imshow(matrix, cmap='Blues', aspect='equal')
    
    # Set ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.set_yticklabels(components)
    
    # Add grid
    ax.set_xticks(np.arange(n+1)-0.5, minor=True)
    ax.set_yticks(np.arange(n+1)-0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Add title and labels
    ax.set_title('Component Dependency Matrix\n(Row depends on Column)', fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Dependencies', fontsize=12)
    ax.set_ylabel('Components', fontsize=12)
    
    # Add text annotations
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == 1:
                ax.text(j, i, '✓', ha='center', va='center', 
                       fontsize=12, weight='bold', color='white')
    
    plt.tight_layout()
    return fig

def create_enhanced_features_diagram():
    """Create a diagram showing where new features fit in"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Before and After comparison
    ax.text(8, 9.5, 'Enhanced Sentiment Analysis: Before vs After', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Before (left side)
    ax.text(4, 8.5, 'BEFORE (Original)', ha='center', va='center', 
            fontsize=14, weight='bold', color='darkred')
    
    before_box = patches.Rectangle((0.5, 6), 7, 2, facecolor='#FFEBEE', edgecolor='darkred', linewidth=2)
    ax.add_patch(before_box)
    
    before_text = """Original System:
• Single source: Polygon.io only
• 3 models: FinBERT, RoBERTa, DistilBERT
• Simple averaging
• No engagement weighting
• No momentum calculations"""
    
    ax.text(4, 7, before_text, ha='center', va='center', fontsize=10)
    
    # After (right side)
    ax.text(12, 8.5, 'AFTER (Enhanced)', ha='center', va='center', 
            fontsize=14, weight='bold', color='darkgreen')
    
    after_box = patches.Rectangle((8.5, 5), 7, 3.5, facecolor='#E8F5E8', edgecolor='darkgreen', linewidth=2)
    ax.add_patch(after_box)
    
    after_text = """Enhanced System:
• 6 sources: Polygon, NewsAPI, Alpha Vantage, 
  FMP, Twitter, Reddit
• Same 3 models + weighted scoring
• Engagement-based weighting
• Source-type weighting (News 70%, Social 30%)
• Momentum & velocity calculations
• Confidence filtering
• Content deduplication"""
    
    ax.text(12, 6.75, after_text, ha='center', va='center', fontsize=10)
    
    # New features integration points
    ax.text(8, 4.5, 'Integration Points for New Features', 
            ha='center', va='center', fontsize=14, weight='bold')
    
    integration_points = [
        ("1. Add new data source", "Implement fetch_[source]_sentiment() function", 2, 3.5),
        ("2. Update main() function", "Add source to data collection loop", 8, 3.5),
        ("3. Configure API keys", "Add credentials to .env file", 14, 3.5),
        ("4. Test integration", "Verify output format compatibility", 2, 2.5),
        ("5. Add source weighting", "Update calculate_weighted_sentiment()", 8, 2.5),
        ("6. Update documentation", "Add to CLAUDE.md and help text", 14, 2.5),
    ]
    
    for title, desc, x, y in integration_points:
        point_box = patches.Rectangle((x-1.5, y-0.4), 3, 0.8, facecolor='#F3E5F5', edgecolor='purple')
        ax.add_patch(point_box)
        ax.text(x, y, f"{title}\n{desc}", ha='center', va='center', fontsize=8, weight='bold')
    
    # Flow arrows
    arrow1 = patches.FancyArrowPatch((7.5, 7), (8.5, 7), 
                                   arrowstyle='->', mutation_scale=20, color='blue', linewidth=3)
    ax.add_patch(arrow1)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create all diagrams
    print("Creating InvBankAI System Architecture Diagrams...")
    
    # Main system diagram
    fig1 = create_system_diagram()
    fig1.savefig('/Users/natwat/Desktop/CPSC_Projects/InvBankAI/system_architecture.png', 
                 dpi=300, bbox_inches='tight')
    print("✅ Main system architecture saved as 'system_architecture.png'")
    
    # Dependency matrix
    fig2 = create_dependency_matrix()
    fig2.savefig('/Users/natwat/Desktop/CPSC_Projects/InvBankAI/dependency_matrix.png', 
                 dpi=300, bbox_inches='tight')
    print("✅ Dependency matrix saved as 'dependency_matrix.png'")
    
    # Enhanced features diagram
    fig3 = create_enhanced_features_diagram()
    fig3.savefig('/Users/natwat/Desktop/CPSC_Projects/InvBankAI/enhanced_features.png', 
                 dpi=300, bbox_inches='tight')
    print("✅ Enhanced features diagram saved as 'enhanced_features.png'")
    
    plt.show()