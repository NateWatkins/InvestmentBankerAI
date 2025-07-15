#!/usr/bin/env python3
"""
Simple File Archiver for InvBankAI
===================================

This is a beginner-friendly file archiving system that:
1. Copies important data files to dated archive folders
2. Keeps track of what gets archived
3. Uses simple Python that anyone can understand
4. Won't break your main functions if something goes wrong

Usage:
    from simple_archiver import archive_file
    archive_file("data/features/AAPL_features.csv")
"""

import os
import shutil
from datetime import datetime
import sys

# Add project root to path for config import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import get_ticker

def archive_file(filename):
    """
    Archive a file by copying it to a dated archive folder.
    
    This function:
    1. Creates an archive folder with today's date
    2. Copies the file to that folder
    3. Adds ticker name and timestamp to the archived filename
    4. Prints what happened so you can see it working
    5. Uses try/except so errors don't break your main code
    
    Args:
        filename (str): Path to the file you want to archive
                       Example: "data/features/AAPL_features.csv"
    
    Returns:
        bool: True if archiving worked, False if it failed
    """
    
    try:
        # Step 1: Check if the file actually exists
        if not os.path.exists(filename):
            print(f"⚠️  Archive skipped: File doesn't exist: {filename}")
            return False
        
        # Step 2: Get the current ticker (like AAPL, TSLA, etc.)
        current_ticker = get_ticker()
        
        # Step 3: Create a date string for today (like "2025-07-15")
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Step 4: Create the archive folder path
        # This will look like: "archive/AAPL/2025-07-15/"
        archive_folder = os.path.join("archive", current_ticker, today)
        
        # Step 5: Create the archive folder if it doesn't exist
        # os.makedirs creates all the folders in the path if needed
        os.makedirs(archive_folder, exist_ok=True)
        print(f"📁 Archive folder ready: {archive_folder}")
        
        # Step 6: Figure out what to name the archived file
        # Get just the filename (not the full path)
        original_filename = os.path.basename(filename)
        
        # Add a timestamp to make the filename unique
        # This prevents overwriting if we archive the same file multiple times
        timestamp = datetime.now().strftime("%H-%M-%S")
        
        # Split the filename into name and extension
        # Example: "AAPL_features.csv" becomes ["AAPL_features", ".csv"]
        name_part, extension_part = os.path.splitext(original_filename)
        
        # Create the new archived filename
        # Example: "AAPL_features_15-30-45.csv"
        archived_filename = f"{name_part}_{timestamp}{extension_part}"
        
        # Step 7: Create the full path for the archived file
        archived_filepath = os.path.join(archive_folder, archived_filename)
        
        # Step 8: Copy the file to the archive location
        # shutil.copy2 copies the file and preserves timestamps
        shutil.copy2(filename, archived_filepath)
        
        # Step 9: Check that the copy worked by seeing if the new file exists
        if os.path.exists(archived_filepath):
            print(f"✅ Archived successfully!")
            print(f"   Original: {filename}")
            print(f"   Archive:  {archived_filepath}")
            return True
        else:
            print(f"❌ Archive failed: Copied file doesn't exist")
            return False
            
    except Exception as error:
        # If anything goes wrong, print the error but don't crash
        print(f"❌ Archive error for {filename}: {error}")
        return False

def archive_multiple_files(file_list):
    """
    Archive multiple files at once.
    
    This is a helper function that calls archive_file() for each file
    in a list. Useful when you want to archive several files together.
    
    Args:
        file_list (list): List of file paths to archive
                         Example: ["file1.csv", "file2.csv", "file3.csv"]
    
    Returns:
        dict: Results showing which files succeeded/failed
    """
    
    results = {
        'successful': [],
        'failed': [],
        'total': len(file_list)
    }
    
    print(f"📦 Starting to archive {len(file_list)} files...")
    
    # Loop through each file in the list
    for i, filename in enumerate(file_list, 1):
        print(f"\n--- Archiving file {i}/{len(file_list)} ---")
        
        # Try to archive this file
        success = archive_file(filename)
        
        # Keep track of results
        if success:
            results['successful'].append(filename)
        else:
            results['failed'].append(filename)
    
    # Print a summary
    print(f"\n📊 Archive Summary:")
    print(f"   ✅ Successful: {len(results['successful'])}")
    print(f"   ❌ Failed: {len(results['failed'])}")
    
    if results['failed']:
        print(f"   Failed files: {results['failed']}")
    
    return results

def clean_old_archives(days_to_keep=30):
    """
    Clean up old archive folders to save disk space.
    
    This function removes archive folders older than the specified number of days.
    It's optional - you can run it occasionally to clean up old files.
    
    Args:
        days_to_keep (int): How many days of archives to keep (default: 30)
    
    Returns:
        int: Number of folders deleted
    """
    
    try:
        print(f"🧹 Cleaning archives older than {days_to_keep} days...")
        
        # Check if archive folder exists
        if not os.path.exists("archive"):
            print("   No archive folder found - nothing to clean")
            return 0
        
        deleted_count = 0
        current_time = datetime.now()
        
        # Loop through each ticker folder in the archive
        for ticker_folder in os.listdir("archive"):
            ticker_path = os.path.join("archive", ticker_folder)
            
            # Skip if it's not a folder
            if not os.path.isdir(ticker_path):
                continue
            
            # Loop through each date folder
            for date_folder in os.listdir(ticker_path):
                date_path = os.path.join(ticker_path, date_folder)
                
                # Skip if it's not a folder
                if not os.path.isdir(date_path):
                    continue
                
                try:
                    # Try to parse the date from the folder name
                    folder_date = datetime.strptime(date_folder, "%Y-%m-%d")
                    
                    # Calculate how old this folder is
                    age_in_days = (current_time - folder_date).days
                    
                    # If it's older than our limit, delete it
                    if age_in_days > days_to_keep:
                        shutil.rmtree(date_path)
                        print(f"   Deleted old archive: {date_path}")
                        deleted_count += 1
                        
                except ValueError:
                    # If we can't parse the date, skip this folder
                    print(f"   Skipped folder with invalid date format: {date_folder}")
                    continue
        
        print(f"✅ Cleanup complete. Deleted {deleted_count} old archive folders.")
        return deleted_count
        
    except Exception as error:
        print(f"❌ Cleanup error: {error}")
        return 0

# Example usage and testing
if __name__ == "__main__":
    """
    This code runs when you execute this file directly.
    It shows examples of how to use the archiving functions.
    """
    
    print("🧪 Testing the Simple Archiver...")
    print("=" * 50)
    
    # Example 1: Archive a single file
    print("\n1. Testing single file archive:")
    example_files = [
        "config.py",  # This should exist
        "nonexistent_file.txt"  # This shouldn't exist
    ]
    
    for test_file in example_files:
        print(f"\nTesting: {test_file}")
        result = archive_file(test_file)
        print(f"Result: {'Success' if result else 'Failed'}")
    
    # Example 2: Archive multiple files
    print("\n\n2. Testing multiple file archive:")
    test_files = [
        "config.py",
        "simple_archiver.py"
    ]
    
    results = archive_multiple_files(test_files)
    print(f"\nMultiple archive results: {results}")
    
    # Example 3: Show what the archive folder looks like
    print("\n\n3. Current archive structure:")
    if os.path.exists("archive"):
        for root, dirs, files in os.walk("archive"):
            level = root.replace("archive", "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print("   No archive folder created yet")
    
    print("\n" + "=" * 50)
    print("🎉 Testing complete!")