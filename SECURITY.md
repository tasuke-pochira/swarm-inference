# Security Policy

## Supported Versions

Currently, the following versions of Swarm Inference are supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Swarm Inference seriously. If you believe you have found a security vulnerability, please report it to us as follows:

1.  **Do not open a public GitHub issue.**
2.  Email your report to **tasuke.pochira@example.com**.
3.  Include as much information as possible, including steps to reproduce.

### Our Response

*   We will acknowledge receipt of your report within 48 hours.
*   We will send a formal response within 7 days, including a timeline for a fix if necessary.
*   We will keep you updated on our progress.

### Scope

This policy applies to all code within this repository. It does not apply to third-party dependencies, although we will assist in reporting issues to those projects where appropriate.

## Secure Best Practices

Users of Swarm Inference are encouraged to:

*   Always run with TLS enabled.
*   Enable mutual authentication (mTLS) for all node communication.
*   Regularly audit logs for suspicious activity.
*   Keep the system updated to the latest supported version.
