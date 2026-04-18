"""
Quick test script to verify adapter system is working.
"""
import sys
sys.path.insert(0, 'backend')

from app.adapters import get_all_adapters, get_available_adapters, get_adapter

print("=" * 60)
print("Testing Evidence Adapter System")
print("=" * 60)

# Test 1: Get all adapters
print("\n1. All registered adapters:")
all_adapters = get_all_adapters()
for name, adapter in all_adapters.items():
    print(f"   - {name}: {adapter.description}")
    print(f"     Available: {adapter.is_available}")

# Test 2: Get available adapters (API format)
print("\n2. Available adapters (API format):")
available = get_available_adapters()
for adapter_info in available:
    print(f"   - {adapter_info}")

# Test 3: Get specific adapter
print("\n3. Get specific adapter:")
arxiv = get_adapter("arxiv")
if arxiv:
    print(f"   ArXiv adapter: {arxiv.name} - {arxiv.description}")
    print(f"   Available: {arxiv.is_available}")
else:
    print("   ERROR: ArXiv adapter not found!")

patents = get_adapter("google_patents")
if patents:
    print(f"   Patents adapter: {patents.name} - {patents.description}")
    print(f"   Available: {patents.is_available}")
else:
    print("   ERROR: Patents adapter not found!")

# Test 4: Test search interface (dry run - no actual API calls)
print("\n4. Testing adapter interface:")
print(f"   ArXiv adapter has search method: {hasattr(arxiv, 'search')}")
print(f"   ArXiv adapter has convert_to_papers method: {hasattr(arxiv, 'convert_to_papers')}")
print(f"   Patents adapter has search method: {hasattr(patents, 'search')}")
print(f"   Patents adapter has convert_to_papers method: {hasattr(patents, 'convert_to_papers')}")

print("\n" + "=" * 60)
print("✓ Adapter system test complete!")
print("=" * 60)
