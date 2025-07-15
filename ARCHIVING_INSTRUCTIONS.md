# Simple File Archiving Instructions

## ✅ What's Already Done

I've added simple file archiving to your main scripts:
- `scripts/FH_getSent.py` ✅
- `scripts/dowload_data.py` ✅
- `scripts/compute_ema.py` ✅
- `scripts/combineSingleDB.py` ✅

## 🚀 How It Works

Every time your scripts save a file, they automatically archive it to:
```
archive/[TICKER]/[DATE]/[filename_timestamp.extension]
```

Example:
```
archive/AAPL/2025-07-14/AAPL_features_15-30-45.csv
archive/AAPL/2025-07-14/AAPL_sentiment_combined_15-32-10.csv
```

## 📝 To Add Archiving to More Scripts

### Step 1: Add the import (top of your script)
```python
# Add these lines after your other imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simple_archiver import archive_file
```

### Step 2: Add ONE line after any file save
```python
# Your existing code that saves a file
df.to_csv("some_file.csv", index=False)
print("File saved!")

# Add this ONE line right after:
archive_file("some_file.csv")
```

That's it! Super simple.

## 🧪 Test the Archiver

Run this to test it:
```bash
python3 simple_archiver.py
```

## 🧹 Clean Old Archives (Optional)

To clean up old archives and save disk space:
```python
from simple_archiver import clean_old_archives

# Keep 30 days of archives (delete older ones)
clean_old_archives(30)
```

## 📁 Archive Folder Structure

```
archive/
├── AAPL/
│   ├── 2025-07-14/
│   │   ├── AAPL_features_09-15-30.csv
│   │   ├── AAPL_sentiment_09-16-45.csv
│   │   └── AAPL_ready_09-18-22.csv
│   └── 2025-07-15/
│       └── AAPL_features_14-22-11.csv
├── TSLA/
│   └── 2025-07-14/
│       └── TSLA_features_10-30-15.csv
└── NVDA/
    └── 2025-07-14/
        └── NVDA_ready_11-45-33.csv
```

## 🛡️ Error Handling

The archiver uses try/except blocks, so:
- ✅ If archiving works: You see success messages
- ⚠️ If archiving fails: You see error messages, but your main script keeps running
- 📁 If folders don't exist: They get created automatically
- 🔍 If files don't exist: Archiving is skipped with a warning

## 🎯 What Gets Archived

Currently archiving these important files:
- Raw price data: `{TICKER}_raw.csv`
- Features data: `{TICKER}_features.csv`
- Sentiment data: `{TICKER}_sentiment_combined.csv`
- Final data: `{TICKER}_ready.csv`
- Merged data: Any output from `combineSingleDB.py`

## 💡 Tips

1. **Check your archives**: Look in the `archive/` folder to see your saved files
2. **Timestamps prevent overwrites**: Each archived file gets a unique timestamp
3. **Ticker-based organization**: Archives are organized by ticker symbol
4. **Simple debugging**: All archive operations print what they're doing
5. **No complexity**: Uses basic Python - no fancy libraries needed

## 🔧 Customizing

To change how archiving works, edit `simple_archiver.py`:
- Change folder structure in `archive_file()` function
- Modify timestamps in the filename creation section
- Adjust error messages in the print statements

Everything is commented so it's easy to understand and modify!