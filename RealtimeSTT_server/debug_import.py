#!/usr/bin/env python3
"""
Debug script to test the AudioToTextRecorder import and backend parameter.
"""

import sys
import os
import inspect

# Add the parent directory to the path to import local RealtimeSTT
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Python path:")
for path in sys.path[:5]:  # Show first 5 paths
    print(f"  {path}")

try:
    # Import directly from the local audio_recorder module
    from RealtimeSTT.audio_recorder import AudioToTextRecorder
    
    print(f"\nSuccessfully imported AudioToTextRecorder from: {AudioToTextRecorder.__module__}")
    print(f"File location: {inspect.getfile(AudioToTextRecorder)}")
    
    # Get the constructor signature
    sig = inspect.signature(AudioToTextRecorder.__init__)
    print(f"\nConstructor signature:")
    print(f"  {sig}")
    
    # Check if backend parameter exists
    if 'backend' in sig.parameters:
        backend_param = sig.parameters['backend']
        print(f"\nBackend parameter found:")
        print(f"  Name: {backend_param.name}")
        print(f"  Default: {backend_param.default}")
        print(f"  Annotation: {backend_param.annotation}")
    else:
        print("\n❌ Backend parameter NOT found in constructor!")
        print("Available parameters:")
        for name, param in sig.parameters.items():
            if name != 'self':
                print(f"  - {name}: {param.annotation} = {param.default}")
    
    # Try to create an instance with backend parameter
    print(f"\nTesting instantiation with backend parameter...")
    try:
        recorder = AudioToTextRecorder(backend="faster_whisper", model="base", use_microphone=False, spinner=False)
        print("✅ Successfully created AudioToTextRecorder with backend parameter!")
        print(f"Backend: {recorder.backend}")
        recorder.shutdown()
    except Exception as e:
        print(f"❌ Failed to create AudioToTextRecorder: {e}")
        
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
