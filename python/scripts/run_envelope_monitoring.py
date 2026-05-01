#!/usr/bin/env python3
"""
Envelope monitoring test runner for CI/CD integration.

This script runs envelope-specific tests with detailed reporting and generates
artifacts for continuous monitoring of envelope module stability and performance.
"""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def run_envelope_tests():
    """Run all envelope-related tests with detailed reporting."""
    print("🔍 Running Envelope Monitoring Test Suite")
    print("=" * 60)

    test_categories = [
        ("basic", "Basic envelope functionality", "tests/test_envelope_monitoring.py::TestEnvelopeBasicBehavior"),
        ("anticlick", "Anti-click protection", "tests/test_envelope_monitoring.py::TestEnvelopeAntiClickProtection"),
        ("performance", "Performance monitoring", "tests/test_envelope_monitoring.py::TestEnvelopePerformance"),
        ("regression", "Regression prevention", "tests/test_envelope_monitoring.py::TestEnvelopeRegressionPrevention"),
        ("audio", "Audio integration", "tests/test_envelope_monitoring.py::TestEnvelopeAudioIntegration"),
    ]

    results = {}
    total_start = time.time()

    for category, description, test_args in test_categories:
        print(f"\n📋 {description}")
        print("-" * 40)

        start_time = time.time()
        cmd = ["python", "-m", "pytest"] + test_args.split() + ["-v", "--tb=short"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per category
            )

            end_time = time.time()
            duration = end_time - start_time

            # Parse pytest output for pass/fail counts
            output_lines = result.stdout.split('\n')
            summary_line = [line for line in output_lines if "passed" in line or "failed" in line]

            status = "PASS" if result.returncode == 0 else "FAIL"

            results[category] = {
                "status": status,
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "summary": summary_line[-1] if summary_line else "No summary available"
            }

            print(f"Status: {status}")
            print(f"Duration: {duration:.2f}s")
            if summary_line:
                print(f"Summary: {summary_line[-1]}")

            if result.returncode != 0:
                print("❌ FAILED - Error output:")
                print(result.stderr)

        except subprocess.TimeoutExpired:
            results[category] = {
                "status": "TIMEOUT",
                "duration": 300,
                "returncode": -1,
                "stdout": "",
                "stderr": "Test timed out after 5 minutes",
                "summary": "TIMEOUT"
            }
            print("⏰ TIMEOUT - Tests took too long")

        except Exception as e:
            results[category] = {
                "status": "ERROR",
                "duration": 0,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "summary": f"Error: {e}"
            }
            print(f"💥 ERROR - {e}")

    total_duration = time.time() - total_start

    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 ENVELOPE MONITORING SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for r in results.values() if r["status"] == "PASS")
    failed_count = sum(1 for r in results.values() if r["status"] == "FAIL")
    timeout_count = sum(1 for r in results.values() if r["status"] == "TIMEOUT")
    error_count = sum(1 for r in results.values() if r["status"] == "ERROR")

    print(f"Total test categories: {len(test_categories)}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {failed_count}")
    print(f"⏰ Timeouts: {timeout_count}")
    print(f"💥 Errors: {error_count}")
    print(f"⏱️  Total duration: {total_duration:.2f}s")

    # Detailed results
    print("\n📋 Detailed Results:")
    for category, result in results.items():
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "TIMEOUT": "⏰",
            "ERROR": "💥"
        }.get(result["status"], "❓")

        print(f"  {status_icon} {category:12s} {result['status']:8s} ({result['duration']:6.2f}s)")
        if result["status"] != "PASS" and result["stderr"]:
            print(f"      Error: {result['stderr'][:100]}...")

    return results, total_duration


def generate_monitoring_report(results, duration):
    """Generate JSON report for CI/CD systems."""
    timestamp = datetime.now().isoformat()

    report = {
        "timestamp": timestamp,
        "total_duration": duration,
        "summary": {
            "total_categories": len(results),
            "passed": sum(1 for r in results.values() if r["status"] == "PASS"),
            "failed": sum(1 for r in results.values() if r["status"] == "FAIL"),
            "timeouts": sum(1 for r in results.values() if r["status"] == "TIMEOUT"),
            "errors": sum(1 for r in results.values() if r["status"] == "ERROR")
        },
        "categories": {}
    }

    for category, result in results.items():
        report["categories"][category] = {
            "status": result["status"],
            "duration": result["duration"],
            "returncode": result["returncode"],
            "summary": result["summary"],
            "has_stderr": bool(result["stderr"])
        }

    # Ensure reports directory exists
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    # Save report
    report_file = reports_dir / "envelope_monitoring_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved: {report_file}")
    return report


def check_critical_failures(results):
    """Check for critical failures that should block deployment."""
    critical_categories = ["basic", "anticlick", "regression"]

    critical_failures = []
    for category in critical_categories:
        if category in results and results[category]["status"] != "PASS":
            critical_failures.append(category)

    if critical_failures:
        print("\n🚨 CRITICAL FAILURES DETECTED:")
        for category in critical_failures:
            print(f"  ❌ {category}: {results[category]['status']}")
        print("\n🛑 DEPLOYMENT SHOULD BE BLOCKED")
        return False
    else:
        print("\n✅ All critical tests passed - safe for deployment")
        return True


def run_performance_benchmarks():
    """Run additional performance benchmarks."""
    print("\n⚡ Running Performance Benchmarks")
    print("-" * 40)

    try:
        # Run performance-specific envelope tests
        cmd = ["python", "-m", "pytest", "tests/test_envelope_monitoring.py::TestEnvelopePerformance", "-v", "--tb=short"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode == 0:
            print("✅ Performance benchmarks passed")
        else:
            print("❌ Performance benchmarks failed")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"💥 Benchmark error: {e}")
        return False


def main():
    """Main monitoring script execution."""
    start_time = time.time()

    try:
        # Run main test suite
        results, duration = run_envelope_tests()

        # Generate monitoring report
        report = generate_monitoring_report(results, duration)

        # Check for critical failures
        deployment_safe = check_critical_failures(results)

        # Run performance benchmarks
        perf_ok = run_performance_benchmarks()

        # Final assessment
        print("\n" + "=" * 60)
        print("🎯 FINAL ASSESSMENT")
        print("=" * 60)

        overall_success = deployment_safe and perf_ok

        if overall_success:
            print("✅ ENVELOPE MODULE: HEALTHY")
            print("   All critical tests passed")
            print("   Performance within acceptable bounds")
            print("   Safe for production deployment")
            exit_code = 0
        else:
            print("❌ ENVELOPE MODULE: ISSUES DETECTED")
            if not deployment_safe:
                print("   Critical test failures detected")
            if not perf_ok:
                print("   Performance issues detected")
            print("   Review required before deployment")
            exit_code = 1

        total_time = time.time() - start_time
        print(f"\n⏱️  Total monitoring time: {total_time:.2f}s")

        return exit_code

    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring interrupted by user")
        return 2
    except Exception as e:
        print(f"\n\n💥 Monitoring script error: {e}")
        import traceback
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
