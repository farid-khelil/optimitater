
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

def is_hex(s):
    try:
        int(s, 16)
        return True
    except:
        return False
    
def load_data(obj):
    idx = input("Select dataset (1: RBA, 2: WPD, 3: PEHF, 4: Custom): ")
    
    if idx == '1' :
        df = pd.read_excel('/home/farid/pfe/data/processed/ransomware/RBA.xlsx')
        target_col = 'Family'
        encodes =  ['EntryPoint', 'PEType', 'magic_number', 'bytes_on_last_page', 'pages_in_file', 'relocations', 'size_of_header', 'min_extra_paragraphs', 'max_extra_paragraphs', 'init_ss_value', 'init_sp_value', 'init_ip_value', 'init_cs_value', 'over_lay_number', 'oem_identifier', 'address_of_ne_header', 'Magic', 'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase', 'SectionAlignment', 'FileAlignment', 'OperatingSystemVersion', 'ImageVersion', 'SizeOfImage', 'SizeOfHeaders', 'Checksum', 'Subsystem', 'SizeofStackReserve', 'SizeofStackCommit', 'SizeofHeapCommit', 'SizeofHeapReserve', 'LoaderFlags', 'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData', 'text_PointerToRawData', 'text_PointerToRelocations', 'text_PointerToLineNumbers', 'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData', 'rdata_PointerToRawData', 'rdata_PointerToRelocations', 'rdata_PointerToLineNumbers', 'rdata_Characteristics']
        drops = ['md5', 'sha1', 'file_extension', 'MachineType', 'DllCharacteristics', 'text_Characteristics', 'Class', 'Category']
    elif idx == '2' :
        df = pd.read_excel('/home/farid/pfe/data/processed/ransomware/WPD.xlsx')
        target_col = 'Benign'
        encodes = []
        drops = ['FileName', 'md5Hash']
    elif idx == '3' :
        df = pd.read_csv('/home/farid/pfe/data/processed/ransomware/PEHF.csv')
        target_col = 'GR'
        encodes = []
        drops = ['filename']
    elif idx == '4':
        # RISS dataset — no header row, last column is the target label
        df = pd.read_csv(obj.data_path, header=None)
        target_col = df.columns[-1]          # integer index of last column
        df[target_col] = df[target_col].round().astype(int)  # fix any float labels (e.g. 1.15 → 1)
        encodes = []
        drops = []
    else:
        print(idx)
        file_path = obj.data_path
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
        print("Columns:")
        mixed_list = df.columns.tolist()
        strings_list = [item for item in mixed_list if isinstance(item, str)]
        print(strings_list)
        target_col = input("Enter target column name: ")
        columns = df.columns.tolist()
        encodes = []
        drops = []
        for col in columns:
            if col != target_col :
                if df[col].dtype == 'string' or df[col].dtype == 'object':
                    print(df[col])
                    tmp = input(f"Is column '{col}' categorical? (y/n): ")
                    if tmp == 'y':
                        encodes.append(col)
                    else:
                        drops.append(col)
        print(f"Encodes: {encodes}")
        print(f"Drops: {drops}")
                  
    

    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if y.dtype == 'object' or y.dtype == 'string':
        print("Encoding target column")
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)

    for enc in encodes:
        le_target = LabelEncoder()
        X[enc] = le_target.fit_transform(X[enc].astype(str) )
    for drop in drops:
        X = X.drop(columns=[drop])
    # for col in columns:
    #     if X[col].apply(is_hex).all():
    #         # Convert entire column to numeric
    #         X[col] = X[col].apply(lambda x: int(x, 16))
    #     if X[col].dtype == 'string' or X[col].dtype == 'object':
    #         print(X[col])
    #         tmp = input(f"Is column '{col}' categorical? (y/n): ")
    #         if tmp.lower() == 'y':
    #             le_target = LabelEncoder()
    #             X[col] = le_target.fit_transform(X[col].astype(str) )
    #             print(X[col])
    #         else:
    #             X = X.drop(columns=[col])
    obj.n_classes = len(np.unique(y))
    class_counts = np.bincount(y)
    stratify_first = y if class_counts.min() >= 2 else None
    X_train_val, obj.X_test, y_train_val, obj.y_test = train_test_split(
        X, y,
        test_size=0.10,
        random_state=42,
        stratify=stratify_first
    )
    class_counts_train = np.bincount(y_train_val)
    stratify_second = y_train_val if class_counts_train.min() >= 2 else None
    obj.X_train, obj.X_val, obj.y_train, obj.y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.2222,
        random_state=42,
        stratify=stratify_second
    )
    
    obj.X_train = obj.scaler.fit_transform(obj.X_train)
    obj.X_val = obj.scaler.transform(obj.X_val)
    obj.X_test = obj.scaler.transform(obj.X_test)

    obj.n_features = obj.X_train.shape[1]

    print("Data loaded successfully.")
    _plot_class_distribution(df[target_col], target_col)


def _plot_class_distribution(y_series, target_col):
    """Save a RedBull-themed bar + donut chart of the target class distribution."""
    RB_BG    = '#0D0D0D'
    RB_PANEL = '#1A1A1A'
    RB_RED   = '#E8292A'
    RB_GOLD  = '#FFD700'
    RB_WHITE = '#F0F0F0'
    RB_GRAY  = '#888888'
    PALETTE  = [
        '#E8292A','#4FC3F7','#69F0AE','#FFD54F','#CE93D8',
        '#FF8A65','#80DEEA','#A5D6A7','#FFF176','#EF9A9A',
        '#B0BEC5','#FFCC80','#80CBC4','#C5E1A5','#F48FB1',
    ]

    counts  = y_series.value_counts().sort_values(ascending=False)
    labels  = counts.index.tolist()
    values  = counts.values
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    total   = values.sum()

    fig, axes = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={'width_ratios': [2, 1]},
        facecolor=RB_BG
    )
    fig.patch.set_facecolor(RB_BG)

    # ── Bar chart ──────────────────────────────────────────────────────────────
    ax_bar = axes[0]
    ax_bar.set_facecolor(RB_PANEL)
    x    = np.arange(len(labels))
    bars = ax_bar.bar(x, values, color=colors, width=0.65, zorder=3,
                      linewidth=0.5, edgecolor='#2a2a2a')
    for bar, val in zip(bars, values):
        pct = val / total * 100
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + values.max() * 0.012,
            f'{val:,}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=7.5,
            color=RB_WHITE, fontweight='bold'
        )
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color=RB_WHITE)
    ax_bar.set_ylabel('Number of Samples', color=RB_WHITE, fontsize=11, labelpad=8)
    ax_bar.set_xlabel(target_col, color=RB_WHITE, fontsize=11, labelpad=8)
    ax_bar.set_title(f'Target Class Distribution — {target_col}',
                     color=RB_GOLD, fontsize=14, fontweight='bold', pad=14)
    ax_bar.tick_params(axis='both', colors=RB_WHITE)
    ax_bar.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f'{int(v):,}')
    )
    ax_bar.spines[:].set_color('#333333')
    ax_bar.grid(axis='y', color='#2e2e2e', linewidth=0.8, zorder=0)
    ax_bar.axhline(values.max(), color=RB_RED, linewidth=0.8,
                   linestyle='--', alpha=0.4, zorder=2)

    # ── Donut chart ────────────────────────────────────────────────────────────
    ax_pie = axes[1]
    ax_pie.set_facecolor(RB_PANEL)
    wedges, _, autotexts = ax_pie.pie(
        values,
        labels=None,
        colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        pctdistance=0.78,
        startangle=140,
        wedgeprops=dict(width=0.55, linewidth=1.2, edgecolor=RB_BG),
    )
    for at in autotexts:
        at.set_color(RB_BG)
        at.set_fontsize(7.5)
        at.set_fontweight('bold')
    ax_pie.text(0, 0, f'{total:,}\nSamples',
                ha='center', va='center', fontsize=11,
                color=RB_WHITE, fontweight='bold')
    ax_pie.set_title('Class Share', color=RB_GOLD, fontsize=13,
                     fontweight='bold', pad=10)
    ax_pie.legend(wedges, labels,
                  loc='lower center', bbox_to_anchor=(0.5, -0.28),
                  ncol=2, fontsize=8, framealpha=0, labelcolor=RB_WHITE)

    fig.text(0.5, 0.01,
             f'{len(labels)} classes  |  {total:,} total samples',
             ha='center', fontsize=9, color=RB_GRAY, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    out_dir  = '/home/farid/pfe/results/figures'
    out_file = os.path.join(out_dir, 'target_class_distribution.png')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_file, dpi=160, bbox_inches='tight', facecolor=RB_BG)
    plt.close()
    print(f'📊  Class distribution chart saved → {out_file}')


