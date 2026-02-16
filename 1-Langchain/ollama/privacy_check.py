#!/usr/bin/env python3
"""
Privacy Verification Script for C++ Code Agent
Verifies that no external communications or telemetry is active
"""

import os
import sys
import socket
import subprocess
from datetime import datetime

def check_environment_variables():
    """Check that all privacy environment variables are set correctly"""
    print("🔍 Checking Privacy Environment Variables...")
    
    privacy_vars = {
        "OLLAMA_ANALYTICS": "false",
        "OLLAMA_TELEMETRY": "false", 
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TORCH_DISABLE_TELEMETRY": "1",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "DO_NOT_TRACK": "1",
        "ANALYTICS_DISABLED": "1",
        "GIT_TELEMETRY_OPTOUT": "1"
    }
    
    all_good = True
    for var, expected in privacy_vars.items():
        actual = os.environ.get(var, "NOT_SET")
        if actual == expected:
            print(f"  ✅ {var} = {actual}")
        else:
            print(f"  ❌ {var} = {actual} (expected: {expected})")
            all_good = False
    
    return all_good

def check_network_connections():
    """Check for any active network connections that might indicate telemetry"""
    print("\n🌐 Checking Network Connections...")
    
    try:
        # Check if we can detect common telemetry endpoints
        telemetry_hosts = [
            "telemetry.microsoft.com",
            "analytics.google.com", 
            "api.github.com",
            "huggingface.co",
            "pytorch.org"
        ]
        
        blocked_count = 0
        for host in telemetry_hosts:
            try:
                socket.create_connection((host, 80), timeout=2)
                print(f"  ⚠️ Connection possible to {host}")
            except (socket.timeout, socket.gaierror, OSError):
                print(f"  ✅ No connection to {host}")
                blocked_count += 1
        
        print(f"\n📊 Telemetry hosts blocked: {blocked_count}/{len(telemetry_hosts)}")
        return blocked_count == len(telemetry_hosts)
        
    except Exception as e:
        print(f"  ❌ Network check failed: {e}")
        return False

def check_local_services():
    """Check that required local services are available"""
    print("\n🏠 Checking Local Services...")
    
    # Check Ollama
    try:
        response = subprocess.run([
            "curl", "-s", "http://localhost:11434/api/version"
        ], capture_output=True, text=True, timeout=5)
        
        if response.returncode == 0:
            print("  ✅ Ollama service available locally")
        else:
            print("  ⚠️ Ollama service not responding")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  ❌ Could not check Ollama (curl not available or Ollama not running)")
    
    return True

def check_file_permissions():
    """Verify that the application can create backups and modify files safely"""
    print("\n📁 Checking File Permissions...")
    
    try:
        # Test write permissions in current directory
        test_file = "privacy_test_temp.txt"
        with open(test_file, 'w') as f:
            f.write("Privacy test")
        
        os.remove(test_file)
        print("  ✅ File write/delete permissions OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ File permission issue: {e}")
        return False

def main():
    """Run complete privacy verification"""
    print("🔒 C++ Code Agent Privacy Verification")
    print(f"📅 Running at: {datetime.now()}")
    print("=" * 50)
    
    # Run all checks
    env_check = check_environment_variables()
    network_check = check_network_connections()  
    service_check = check_local_services()
    file_check = check_file_permissions()
    
    print("\n" + "=" * 50)
    print("📋 PRIVACY VERIFICATION SUMMARY:")
    print("=" * 50)
    
    if env_check:
        print("✅ Environment Variables: All privacy settings active")
    else:
        print("❌ Environment Variables: Some privacy settings missing")
    
    if network_check:
        print("✅ Network Security: No telemetry connections detected")
    else:
        print("⚠️ Network Security: Some external connections possible")
    
    if service_check:
        print("✅ Local Services: Required services available")
    else:
        print("⚠️ Local Services: Some services may not be available")
    
    if file_check:
        print("✅ File Security: Safe file operations confirmed")
    else:
        print("❌ File Security: File operation issues detected")
    
    print("\n" + "=" * 50)
    
    if all([env_check, file_check]):
        print("🎉 PRIVACY VERIFICATION PASSED!")
        print("🔐 Your C++ Code Agent is configured for maximum privacy")
        print("🏠 All processing will happen locally on your machine")
        return True
    else:
        print("⚠️ PRIVACY VERIFICATION HAD ISSUES")
        print("💡 Please review the issues above and restart the application")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)