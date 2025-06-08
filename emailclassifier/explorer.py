def explore_gmail_data(emails):
    print(f"Total emails: {len(emails)}")
    
    # Check data quality
    empty_subjects = sum(1 for e in emails if not e['subject'] or e['subject'].strip() == '')
    empty_bodies = sum(1 for e in emails if not e['body'] or e['body'].strip() == '')
    
    print(f"Empty subjects: {empty_subjects}")
    print(f"Empty bodies: {empty_bodies}")
    
    # Analyze sender patterns (important for importance detection)
    senders = [e['sender'] for e in emails]
    unique_senders = len(set(senders))
    print(f"Unique senders: {unique_senders}")
    
    # Look at email variety for labeling
    print("\n=== Sample subjects (diversity check) ===")
    for i, email in enumerate(emails[:10]):
        print(f"{i+1}. From: {email['sender'][:30]}")
        print(f"    Subject: {email['subject'][:60]}")
        print()
    
    # Check if we have enough data for ML
    if len(emails) < 100:
        print("⚠️  Warning: Need 500+ emails for good ML training")
    else:
        print(f"✓ Good dataset size for ML training")
