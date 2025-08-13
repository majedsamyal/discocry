def explore_gmail_data(emails):
    print(f"Total emails: {len(emails)}")
    
    # Check for empty/missing data
    empty_subjects = sum(1 for e in emails if not e['subject'] or e['subject'].strip() == '')
    empty_bodies = sum(1 for e in emails if not e['body'] or e['body'].strip() == '')
    no_subject_pattern = sum(1 for e in emails if e['subject'].strip().lower() in ['(no subject)', 'no subject'])
    
    print(f"Empty subjects: {empty_subjects}")
    print(f"'(no subject)' emails: {no_subject_pattern}")
    print(f"Empty bodies: {empty_bodies}")
    
    # Analyze sender patterns (important for importance detection)
    senders = [e['sender'] for e in emails]
    unique_senders = len(set(senders))
    print(f"Unique senders: {unique_senders}")
    
    # Email body length stats
    body_lengths = [len(e['body']) for e in emails if e['body']]
    if body_lengths:
        avg_length = sum(body_lengths) / len(body_lengths)
        print(f"\nAverage body length: {avg_length:.0f} characters")
        print(f"Shortest email: {min(body_lengths)} chars")
        print(f"Longest email: {max(body_lengths)} chars")
    
    # Sample emails for manual importance assessment
    print("\n=== Sample emails for importance patterns ===")
    for i, email in enumerate(emails[:10]):
        print(f"{i+1}. From: {email['sender'][:30]}")
        print(f"    Subject: {email['subject'][:60]}")
        print()
    
    # Check if we have enough data for ML
    if len(emails) < 100:
        print("⚠️  Warning: Need 500+ emails for good ML training")
    else:
        print(f"✓ Good dataset size for ML training")
