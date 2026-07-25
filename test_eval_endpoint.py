#!/usr/bin/env python3
"""
Quick test script to verify the FastAPI /eval endpoint is working.
Run this after starting ui.py to ensure the evaluation endpoint is accessible.
"""

import requests
import time
import sys
import pytest

def test_eval_endpoint():
    """Test the /eval endpoint with a simple query"""
    
    EVAL_URL = "http://127.0.0.1:8001"
    TEST_PAYLOAD = {
        "query": "Hello world",
        "chunks": 3,
        "rerank": "false",
        "hybrid": "false",
        "ablation": "V1",
        "rewriting_strategy": "none"
    }
    
    print("=" * 60)
    print("🧪 FastAPI /eval Endpoint Test")
    print("=" * 60)
    print(f"Target URL: {EVAL_URL}/eval")
    print(f"Payload: {TEST_PAYLOAD}\n")
    
    # Test 1: Health check
    print("Test 1️⃣: Health check on root endpoint...")
    try:
        response = requests.get(f"{EVAL_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is healthy: {response.json()}\n")
        else:
            print(f"⚠️  Health check returned {response.status_code}\n")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {EVAL_URL}")
        print("   Make sure ui.py is running: streamlit run ui.py\n")
        pytest.skip(f"Server not running at {EVAL_URL}; skipping integration check")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        assert False, f"Health check failed: {e}"
    
    # Test 2: Query /eval endpoint
    print("Test 2️⃣: Querying /eval endpoint...")
    try:
        response = requests.post(
            f"{EVAL_URL}/eval",
            json=TEST_PAYLOAD,
            timeout=30
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Response: {result}")
            
            if result.get("status") == "success":
                print(f"✅ /eval endpoint working!")
                print(f"   Retrieved {len(result.get('retrieved_context_keys', []))} context keys")
                
                if not result.get('retrieved_context_keys'):
                    print("   ⚠️  Note: No documents retrieved (documents may not be indexed yet)")
            else:
                print(f"❌ Error from endpoint: {result.get('message')}")
                assert False, f"Eval endpoint returned error status: {result.get('message')}"
        else:
            print(f"❌ Server returned {response.status_code}: {response.text[:200]}")
            assert False, f"Eval endpoint HTTP {response.status_code}: {response.text[:200]}"
            
    except requests.exceptions.Timeout:
        print(f"❌ Request timed out after 30 seconds")
        print("   The endpoint may be processing or there may be a server issue")
        assert False, "Eval endpoint timed out after 30 seconds"
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to /eval endpoint at {EVAL_URL}/eval")
        assert False, f"Cannot connect to /eval endpoint at {EVAL_URL}/eval"
    except Exception as e:
        print(f"❌ Error: {e}")
        assert False, f"Eval endpoint request failed: {e}"
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! The /eval endpoint is working.")
    print("=" * 60)
    print("\nYou can now run: python docs/3.0/script.py")
    assert True

if __name__ == "__main__":
    success = test_eval_endpoint()
    sys.exit(0 if success else 1)
