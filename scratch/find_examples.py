import csv

csv_path = r"c:\Users\Nesty\OneDrive\Documents\GitHub\PROJECT-BIDAT\dataset\online_retail.csv"

# We will collect examples for:
# 1. Row with leading/trailing whitespaces (trimming needed)
# 2. Row with cancel transaction (InvoiceNo starts with 'C')
# 3. Row with invalid/unparsable InvoiceDate (if any) or normal format
# 4. Row with Quantity <= 0 or UnitPrice <= 0
# 5. Row with CustomerID null or empty
# 6. Clean/normal row that survives all cleaning

examples = {
    "trim": [],
    "cancel": [],
    "invalid_date": [],
    "invalid_values": [],
    "null_customer": [],
    "normal": []
}

with open(csv_path, mode='r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    print("CSV Fields:", reader.fieldnames)
    
    for i, row in enumerate(reader):
        # 1. Trim check: check if any string column has leading/trailing spaces
        has_whitespace = False
        for col in ["InvoiceNo", "StockCode", "Description", "CustomerID", "Country", "InvoiceDate"]:
            val = row.get(col, "")
            if val and (val.startswith(" ") or val.endswith(" ")):
                has_whitespace = True
                break
        
        # Parse invoice date pattern
        # Standard: yyyy-MM-dd HH:mm:ss, e.g. "2010-12-01 08:26:00"
        date_str = row.get("InvoiceDate", "")
        is_valid_date = True
        # Simple format check (length and separators)
        if len(date_str) < 19 or "-" not in date_str or ":" not in date_str:
            is_valid_date = False
            
        # Cancel check
        invoice_no = row.get("InvoiceNo", "")
        is_cancel = invoice_no.startswith("C")
        
        # Quantity and UnitPrice
        try:
            qty = int(row.get("Quantity", 0))
        except ValueError:
            qty = 0
            
        try:
            price = float(row.get("UnitPrice", 0))
        except ValueError:
            price = 0.0
            
        is_invalid_values = (qty <= 0 or price <= 0)
        
        # Customer ID null or empty
        cust_id = row.get("CustomerID", "")
        is_null_cust = not cust_id or cust_id.strip() == ""
        
        # Save examples
        if has_whitespace and len(examples["trim"]) < 3:
            examples["trim"].append(dict(row))
            
        if is_cancel and len(examples["cancel"]) < 3:
            examples["cancel"].append(dict(row))
            
        if not is_valid_date and len(examples["invalid_date"]) < 3:
            examples["invalid_date"].append(dict(row))
            
        if is_invalid_values and len(examples["invalid_values"]) < 3:
            examples["invalid_values"].append(dict(row))
            
        if is_null_cust and len(examples["null_customer"]) < 3:
            examples["null_customer"].append(dict(row))
            
        # A normal row survives all cleaning:
        if (not has_whitespace and 
            is_valid_date and 
            not is_cancel and 
            qty > 0 and 
            price > 0 and 
            not is_null_cust and 
            len(examples["normal"]) < 3):
            examples["normal"].append(dict(row))
            
        # Stop early if we have enough examples
        if (len(examples["trim"]) >= 3 and 
            len(examples["cancel"]) >= 3 and 
            len(examples["invalid_values"]) >= 3 and 
            len(examples["null_customer"]) >= 3 and 
            len(examples["normal"]) >= 3 and
            i > 50000): # scan enough rows to find invalid date if any
            break

print("\n--- RESULTS ---")
for key, vals in examples.items():
    print(f"\nCategory: {key.upper()} (Found: {len(vals)})")
    for row in vals:
        # Print with repr to show whitespaces
        print({k: repr(v) for k, v in row.items()})
