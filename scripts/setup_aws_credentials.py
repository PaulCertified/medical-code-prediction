#!/usr/bin/env python
"""
Script to set up AWS credentials for the medical code prediction project.
"""
import os
import sys
import argparse
import configparser
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Set up AWS credentials for the medical code prediction project.')
    parser.add_argument('--access-key', type=str, required=True, help='AWS access key ID')
    parser.add_argument('--secret-key', type=str, required=True, help='AWS secret access key')
    parser.add_argument('--region', type=str, default='us-west-2', help='AWS region')
    parser.add_argument('--profile', type=str, default='default', help='AWS profile name')
    
    return parser.parse_args()

def setup_credentials(access_key, secret_key, region, profile):
    """
    Set up AWS credentials in the ~/.aws/credentials file.
    
    Args:
        access_key: AWS access key ID
        secret_key: AWS secret access key
        region: AWS region
        profile: AWS profile name
    """
    # Create the ~/.aws directory if it doesn't exist
    aws_dir = Path.home() / '.aws'
    aws_dir.mkdir(exist_ok=True)
    
    # Create or update the credentials file
    credentials_path = aws_dir / 'credentials'
    config_path = aws_dir / 'config'
    
    # Create or update the credentials file
    credentials = configparser.ConfigParser()
    
    if credentials_path.exists():
        credentials.read(credentials_path)
    
    if profile not in credentials:
        credentials[profile] = {}
    
    credentials[profile]['aws_access_key_id'] = access_key
    credentials[profile]['aws_secret_access_key'] = secret_key
    
    with open(credentials_path, 'w') as f:
        credentials.write(f)
    
    # Create or update the config file
    config = configparser.ConfigParser()
    
    if config_path.exists():
        config.read(config_path)
    
    profile_section = f'profile {profile}' if profile != 'default' else 'default'
    
    if profile_section not in config:
        config[profile_section] = {}
    
    config[profile_section]['region'] = region
    
    with open(config_path, 'w') as f:
        config.write(f)
    
    print(f"AWS credentials set up successfully for profile '{profile}'")
    print(f"Region: {region}")
    print(f"Credentials file: {credentials_path}")
    print(f"Config file: {config_path}")

def main():
    """Main function."""
    args = parse_args()
    
    # Set up AWS credentials
    setup_credentials(
        access_key=args.access_key,
        secret_key=args.secret_key,
        region=args.region,
        profile=args.profile
    )
    
    print("\nYou can now use the AWS CLI and boto3 with these credentials.")
    print("To use a specific profile, set the AWS_PROFILE environment variable:")
    print(f"export AWS_PROFILE={args.profile}")

if __name__ == "__main__":
    main() 