.PHONY: all ci fmt fmt-check lint test test-release doc doc-check check build clean

# Default target
all: ci

# CI target - runs all checks that CI performs
ci: fmt-check lint test test-release doc-check check
	@echo "All CI checks passed!"

# Format code
fmt:
	cargo fmt --all

# Check formatting (CI mode - fails if not formatted)
fmt-check:
	cargo fmt --all -- --check

# Run clippy lints
lint:
	cargo clippy --all-features -- -D warnings

# Run tests
test:
	cargo test --all-features

# Run tests in release mode
test-release:
	cargo test --all-features --release

# Build documentation
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Check documentation (CI mode)
doc-check:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Check compilation (used for MSRV)
check:
	cargo check --all-features

# Build the project
build:
	cargo build --all-features

# Build in release mode
build-release:
	cargo build --all-features --release

# Clean build artifacts
clean:
	cargo clean

# Fix formatting and lint issues automatically where possible
fix: fmt
	cargo clippy --all-features --fix --allow-dirty --allow-staged
