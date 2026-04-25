
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os


def _read_excel_safe(path):
    """Read Excel files reliably on Kaggle/local by forcing openpyxl."""
    return pd.read_excel(path, engine='openpyxl')


def _normalize_columns(df):
    """Strip whitespace / BOM artifacts from column names."""
    df = df.copy()
    df.columns = [str(col).replace('\ufeff', '').strip() for col in df.columns]
    return df


def _resolve_target_column(df, target_name):
    """Resolve a target column by exact then normalized name match."""
    if target_name in df.columns:
        return target_name

    normalized = {str(col).replace('\ufeff', '').strip().lower(): col for col in df.columns}
    return normalized.get(str(target_name).strip().lower())

def is_hex(s):
    try:
        int(s, 16)
        return True
    except:
        return False
    
def load_data(obj, idx=None):
    # idx can be passed directly (required on Kaggle / non-interactive env).
    # Falls back to an interactive prompt when running locally.
    if idx is None:
        idx = input("Select dataset (1: RBA, 2: WPD, 3: PEHF, 4: RISS): ").strip()
    
    if idx == '1' :
        df = _normalize_columns(_read_excel_safe(obj.data_path))
        target_col = _resolve_target_column(df, 'Class')
        encodes =  ['EntryPoint', 'PEType', 'magic_number', 'bytes_on_last_page', 'pages_in_file', 'relocations', 'size_of_header', 'min_extra_paragraphs', 'max_extra_paragraphs', 'init_ss_value', 'init_sp_value', 'init_ip_value', 'init_cs_value', 'over_lay_number', 'oem_identifier', 'address_of_ne_header', 'Magic', 'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase', 'SectionAlignment', 'FileAlignment', 'OperatingSystemVersion', 'ImageVersion', 'SizeOfImage', 'SizeOfHeaders', 'Checksum', 'Subsystem', 'SizeofStackReserve', 'SizeofStackCommit', 'SizeofHeapCommit', 'SizeofHeapReserve', 'LoaderFlags', 'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData', 'text_PointerToRawData', 'text_PointerToRelocations', 'text_PointerToLineNumbers', 'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData', 'rdata_PointerToRawData', 'rdata_PointerToRelocations', 'rdata_PointerToLineNumbers', 'rdata_Characteristics']
        drops = ['md5', 'sha1', 'file_extension', 'MachineType', 'DllCharacteristics', 'text_Characteristics', 'Category', 'Family', ]
    elif idx == '2' :
        df = _normalize_columns(_read_excel_safe(obj.data_path))
        target_col = _resolve_target_column(df, 'Benign')
        encodes = []
        drops = ['FileName', 'md5Hash']
    elif idx == '3' :
        df = pd.read_csv(obj.data_path)
        target_col = 'GR'
        encodes = []
        drops = ['ID','filename']
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
            df = _normalize_columns(_read_excel_safe(file_path))
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
                  
    

    
    if target_col is None or target_col not in df.columns:
        raise KeyError(f"Target column not found. Resolved target={target_col}. Available columns: {list(df.columns)}")

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
        if drop in X.columns:
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
        test_size=0.20,
        random_state=42,
        stratify=stratify_first
    )
    # Keep a persistent 80% training pool (for optional CV) and 20% test holdout.
    obj.X_train_val = np.asarray(X_train_val)
    obj.y_train_val = np.asarray(y_train_val).astype(int)

    # Fit preprocessing on the full training pool only (never on test).
    obj.X_train_val = obj.scaler.fit_transform(obj.X_train_val)
    obj.X_test = obj.scaler.transform(obj.X_test)
    obj.X_train_val = obj.smootheringScaler.fit_transform(obj.X_train_val)
    obj.X_test = obj.smootheringScaler.transform(obj.X_test)

    # ── Feature selection for RISS (16 380 features → manageable subset) ────────
    # Without this the model has ~1 M params for 1 067 samples → extreme overfit.
    if idx == '4':
        # 1. Remove constant / near-constant features
        vt = VarianceThreshold(threshold=0.0)
        obj.X_train_val = vt.fit_transform(obj.X_train_val)
        obj.X_test = vt.transform(obj.X_test)
        print(f"After VarianceThreshold: {obj.X_train_val.shape[1]} features remaining")

        # 2. Keep top-500 features ranked by ANOVA F-score (fit on train only)
        K = min(500, obj.X_train_val.shape[1])
        selector = SelectKBest(f_classif, k=K)
        obj.X_train_val = selector.fit_transform(obj.X_train_val, obj.y_train_val)
        obj.X_test = selector.transform(obj.X_test)
        obj.feature_selector = selector          # store for later use
        print(f"After SelectKBest(k={K}): {obj.X_train_val.shape[1]} features")

    # CV setup on the 80% training pool (default: 5 folds => 80/20 train/val per fold).
    cv_folds = max(1, int(getattr(obj, 'cv_folds', 5)))
    obj.use_cv = bool(getattr(obj, 'use_cv', True) and cv_folds > 1)

    if obj.use_cv:
        counts = np.bincount(obj.y_train_val)
        can_stratify = counts.min() >= cv_folds
        splitter = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            if can_stratify
            else KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        )
        obj.cv_indices = list(splitter.split(obj.X_train_val, obj.y_train_val))

        # Keep first fold as active train/val for legacy paths that expect these attrs.
        train_idx, val_idx = obj.cv_indices[0]
        obj.X_train = obj.X_train_val[train_idx]
        obj.y_train = obj.y_train_val[train_idx]
        obj.X_val = obj.X_train_val[val_idx]
        obj.y_val = obj.y_train_val[val_idx]
        print(f"Cross-validation enabled: {cv_folds} folds on 80% training pool.")
    else:
        class_counts_train = np.bincount(obj.y_train_val)
        stratify_second = obj.y_train_val if class_counts_train.min() >= 2 else None
        obj.X_train, obj.X_val, obj.y_train, obj.y_val = train_test_split(
            obj.X_train_val, obj.y_train_val,
            # 25% of 80% => 20% total as validation
            test_size=0.25,
            random_state=42,
            stratify=stratify_second
        )

    obj.X_train = np.asarray(obj.X_train)
    obj.X_val = np.asarray(obj.X_val)
    obj.X_test = np.asarray(obj.X_test)
    obj.y_train = np.asarray(obj.y_train).astype(int)
    obj.y_val = np.asarray(obj.y_val).astype(int)
    obj.y_test = np.asarray(obj.y_test).astype(int)

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

    out_dir  = '/kaggle/working/figures' if os.path.isdir('/kaggle') else '/home/farid/pfe/results/figures'
    out_file = os.path.join(out_dir, 'target_class_distribution.png')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_file, dpi=160, bbox_inches='tight', facecolor=RB_BG)
    plt.close()
    print(f'📊  Class distribution chart saved → {out_file}')


