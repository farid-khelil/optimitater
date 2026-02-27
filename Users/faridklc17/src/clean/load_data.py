
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split

def is_hex(s):
    try:
        int(s, 16)
        return True
    except:
        return False
    
def load_data(obj):
    idx = input('inter 1 for dataset : ')
    
    if idx == '1' :
        df = pd.read_excel('/home/azureuser/cloudfiles/code/Users/faridklc17/src/RBA.xlsx')
        target_col = 'Family'
        encodes =  ['EntryPoint', 'PEType', 'magic_number', 'bytes_on_last_page', 'pages_in_file', 'relocations', 'size_of_header', 'min_extra_paragraphs', 'max_extra_paragraphs', 'init_ss_value', 'init_sp_value', 'init_ip_value', 'init_cs_value', 'over_lay_number', 'oem_identifier', 'address_of_ne_header', 'Magic', 'SizeOfCode', 'SizeOfInitializedData', 'SizeOfUninitializedData', 'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData', 'ImageBase', 'SectionAlignment', 'FileAlignment', 'OperatingSystemVersion', 'ImageVersion', 'SizeOfImage', 'SizeOfHeaders', 'Checksum', 'Subsystem', 'SizeofStackReserve', 'SizeofStackCommit', 'SizeofHeapCommit', 'SizeofHeapReserve', 'LoaderFlags', 'text_VirtualSize', 'text_VirtualAddress', 'text_SizeOfRawData', 'text_PointerToRawData', 'text_PointerToRelocations', 'text_PointerToLineNumbers', 'rdata_VirtualSize', 'rdata_VirtualAddress', 'rdata_SizeOfRawData', 'rdata_PointerToRawData', 'rdata_PointerToRelocations', 'rdata_PointerToLineNumbers', 'rdata_Characteristics']
        drops = ['md5', 'sha1', 'file_extension', 'MachineType', 'DllCharacteristics', 'text_Characteristics', 'Class', 'Category']
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
    
        print(df.columns.tolist())
        target_col = input("Enter target column name: ")
        problem_type = input("Classification or Regression? (c/r): ")
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
    print("Columns in the dataset:")
    
    print("Data loaded successfully.")


