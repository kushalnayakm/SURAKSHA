import pandas as pd
import numpy as np
import os

def preprocess_real_aadhaar_data():
    """
    Data cleaning and preprocessing for REAL Aadhaar enrollment data
    """
    
    print("=" * 60)
    print("STEP 1: PREPROCESSING REAL AADHAAR ENROLLMENT DATA")
    print("=" * 60)
    
    os.makedirs('data', exist_ok=True)
    
    # LOAD DATA
    print("\n📂 Loading real Aadhaar enrollment data...")
    try:
        df = pd.read_csv('data/real_aadhaar_enrollment_data.csv')
        initial_count = len(df)
        print(f"   ✅ Loaded {initial_count:,} records")
    except FileNotFoundError:
        print("   ❌ Error: File not found!")
        print("   Please run: python load_real_aadhaar_data.py first")
        return None
    
    # STEP 1: CHECK DATA TYPES
    print("\n🔄 Converting data types...")
    df['anonymized_uid'] = df['anonymized_uid'].astype('int64')
    df['operator_id'] = df['operator_id'].astype('int64')
    df['center_id'] = df['center_id'].astype('int64')
    df['document_quality_score'] = df['document_quality_score'].astype('int64')
    df['enrollment_timestamp'] = pd.to_datetime(df['enrollment_timestamp'])
    df['risk_score'] = df['risk_score'].astype('float64')
    print("   ✅ Data types converted")
    
    # STEP 2: CHECK FOR MISSING VALUES
    print("\n🔍 Checking for missing values...")
    missing_before = df.isnull().sum().sum()
    print(f"   Missing values: {missing_before}")
    
    # Remove rows with any missing critical data
    df = df.dropna(subset=['anonymized_uid', 'operator_id', 'center_id', 'enrollment_timestamp', 'biometric_hash'])
    removed_missing = initial_count - len(df)
    print(f"   Records removed: {removed_missing}")
    
    # STEP 3: REMOVE DUPLICATES
    print("\n🔄 Removing duplicates...")
    df_before_dup = len(df)
    df = df.drop_duplicates(subset=['anonymized_uid', 'enrollment_timestamp'])
    removed_duplicates = df_before_dup - len(df)
    print(f"   Duplicate records removed: {removed_duplicates}")
    
    # STEP 4: EXTRACT TEMPORAL FEATURES
    print("\n⏰ Extracting temporal features...")
    df['year'] = df['enrollment_timestamp'].dt.year
    df['month'] = df['enrollment_timestamp'].dt.month
    df['day'] = df['enrollment_timestamp'].dt.day
    df['hour'] = df['enrollment_timestamp'].dt.hour
    df['dayofweek'] = df['enrollment_timestamp'].dt.dayofweek
    print("   ✅ Temporal features extracted")
    
    # STEP 5: GEOGRAPHIC ANALYSIS
    print("\n🗺️  Analyzing geographic patterns...")
    
    # Operators per state (enrollment agency density)
    state_operators = df.groupby('state')['operator_id'].nunique()
    df['operators_in_state'] = df['state'].map(state_operators)
    
    # Centers per district
    district_centers = df.groupby(['state', 'district'])['center_id'].nunique()
    df['centers_in_district'] = df.apply(
        lambda row: district_centers.get((row['state'], row['district']), 0),
        axis=1
    )
    
    print(f"   ✅ Geographic features added")
    print(f"   States: {df['state'].nunique()}")
    print(f"   Districts: {df['district'].nunique()}")
    
    # STEP 6: ENROLLMENT VOLUME ANALYSIS
    print("\n📊 Analyzing enrollment volumes...")
    
    # Daily enrollments per operator
    df['enrollments_per_day'] = df.groupby(['operator_id', 'day'])['anonymized_uid'].transform('count')
    
    # Total enrollments per operator
    operator_totals = df.groupby('operator_id')['anonymized_uid'].transform('count')
    df['operator_total_enrollments'] = operator_totals
    
    # Enrollments per center
    center_enrollments = df.groupby('center_id')['anonymized_uid'].transform('count')
    df['center_total_enrollments'] = center_enrollments
    
    print(f"   ✅ Volume metrics calculated")
    
    # STEP 7: DETECT TEMPORAL ANOMALIES
    print("\n🚨 Detecting temporal anomalies...")
    
    # Burst detection: unusually high enrollment in short time window
    df['is_temporal_burst'] = False
    
    for operator_id in df['operator_id'].unique():
        operator_data = df[df['operator_id'] == operator_id]
        
        # Check for more than 20 enrollments in same hour
        hourly_counts = operator_data.groupby(['day', 'hour']).size()
        burst_threshold = hourly_counts.quantile(0.95) if len(hourly_counts) > 0 else 20
        
        burst_hours = hourly_counts[hourly_counts > burst_threshold].index
        
        for day, hour in burst_hours:
            df.loc[(df['operator_id'] == operator_id) & 
                   (df['day'] == day) & 
                   (df['hour'] == hour), 'is_temporal_burst'] = True
    
    bursts_detected = df['is_temporal_burst'].sum()
    print(f"   Temporal bursts detected: {bursts_detected:,}")
    
    # STEP 8: DETECT BIOMETRIC ANOMALIES
    print("\n🔐 Detecting biometric anomalies...")
    
    # Biometric hash collisions (same biometric for multiple UIDs = FRAUD!)
    biometric_counts = df['biometric_hash'].value_counts()
    duplicate_biometrics = biometric_counts[biometric_counts > 1].index.tolist()
    df['has_biometric_collision'] = df['biometric_hash'].isin(duplicate_biometrics)
    
    collisions = df['has_biometric_collision'].sum()
    print(f"   Biometric hash collisions: {collisions:,}")
    
    # STEP 9: GEOGRAPHIC CONCENTRATION ANALYSIS
    print("\n🎯 Analyzing geographic concentration...")
    
    # Calculate entropy for operator-district distribution
    # High concentration = suspicious
    operator_district = df.groupby('operator_id')['district'].nunique()
    operator_total = df.groupby('operator_id').size()
    
    # Concentration ratio = distinct districts / total enrollments
    concentration_ratio = operator_district / operator_total
    df['operator_concentration'] = df['operator_id'].map(concentration_ratio)
    
    suspicious_concentration = (df['operator_concentration'] < 0.1).sum()
    print(f"   Operators with high concentration: {suspicious_concentration:,}")
    
    # STEP 10: UPDATE RISK SCORES
    print("\n⚠️  Recalculating comprehensive risk scores...")
    
    # Start fresh
    df['final_risk_score'] = 0.0
    
    # Risk factor 1: Biometric collisions (HIGH RISK)
    df.loc[df['has_biometric_collision'], 'final_risk_score'] += 0.4
    
    # Risk factor 2: Temporal bursts
    df.loc[df['is_temporal_burst'], 'final_risk_score'] += 0.3
    
    # Risk factor 3: High concentration
    df.loc[df['operator_concentration'] < 0.1, 'final_risk_score'] += 0.2
    
    # Risk factor 4: Unusual document quality
    quality_threshold = df['document_quality_score'].quantile(0.05)
    df.loc[df['document_quality_score'] < quality_threshold, 'final_risk_score'] += 0.1
    
    # Normalize
    df['final_risk_score'] = np.clip(df['final_risk_score'], 0, 1)
    
    print(f"   Mean risk score: {df['final_risk_score'].mean():.4f}")
    print(f"   High-risk records: {(df['final_risk_score'] > 0.5).sum():,}")
    
    # STEP 11: SAVE PROCESSED DATA
    print("\n💾 Saving processed data...")
    
    # Save full processed dataset
    df.to_csv('data/processed_real_aadhaar_data.csv', index=False)
    print("   ✅ Saved: data/processed_real_aadhaar_data.csv")
    
    # Save high-risk records for investigation
    high_risk = df[df['final_risk_score'] > 0.5]
    high_risk.to_csv('data/flagged_high_risk_records.csv', index=False)
    print(f"   ✅ Saved: data/flagged_high_risk_records.csv ({len(high_risk):,} records)")
    
    # Save by state (for state-wise analysis)
    for state in df['state'].unique():
        state_data = df[df['state'] == state]
        safe_state_name = state.replace('/', '_').replace(' ', '_')
        state_data.to_csv(f'data/state_{safe_state_name}.csv', index=False)
    
    print(f"   ✅ Saved: {df['state'].nunique()} state-wise files")
    
    # STEP 12: SUMMARY REPORT
    print("\n" + "=" * 60)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Data Quality Metrics:")
    print(f"   Initial Records:        {initial_count:,}")
    print(f"   Final Records:          {len(df):,}")
    print(f"   Data Retention:         {(len(df)/initial_count)*100:.2f}%")
    print(f"   Duplicates Removed:     {removed_duplicates:,}")
    
    print(f"\n🗺️  Geographic Coverage:")
    print(f"   States:                 {df['state'].nunique()}")
    print(f"   Districts:              {df['district'].nunique()}")
    print(f"   Enrollment Centers:     {df['center_id'].nunique()}")
    print(f"   Enrollment Operators:   {df['operator_id'].nunique()}")
    
    print(f"\n👥 Age Distribution (Total):")
    print(f"   Age 0-5:                {df['age_0_5'].sum():,}")
    print(f"   Age 5-17:               {df['age_5_17'].sum():,}")
    print(f"   Age 18+:                {df['age_18_greater'].sum():,}")
    
    print(f"\n🚨 Anomalies Detected:")
    print(f"   Temporal Bursts:        {bursts_detected:,}")
    print(f"   Biometric Collisions:   {collisions:,}")
    print(f"   High-Risk Records:      {len(high_risk):,} ({(len(high_risk)/len(df))*100:.2f}%)")
    
    print(f"\n📋 New Features Created:  13")
    print(f"   Temporal Features:      5 (year, month, day, hour, dayofweek)")
    print(f"   Geographic Features:    3 (operators_in_state, centers_in_district, etc.)")
    print(f"   Volume Features:        3 (enrollments per day, operator totals, center totals)")
    print(f"   Anomaly Flags:          2 (temporal_burst, biometric_collision)")
    
    print(f"\n" + "=" * 60)
    
    return df


if __name__ == "__main__":
    df = preprocess_real_aadhaar_data()
    
    if df is not None:
        print(f"\n✅ STEP 1 COMPLETE")
        print(f"\n🔄 Next steps:")
        print(f"   2. Run: python code2_graph_construction.py")
        print(f"   3. Run: python code3_rgcn_training.py")
        print(f"   4. Run: python code4_fraud_detection.py")