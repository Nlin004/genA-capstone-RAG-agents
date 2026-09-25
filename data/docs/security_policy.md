# Company Security Policy

## Purpose

This policy defines the minimum security standards that all employees,
contractors, and systems must follow to protect company and customer data.

## Access Control

Access to production systems is granted on a least-privilege basis. Every
employee is assigned a unique account; shared accounts are prohibited.
Multi-factor authentication (MFA) is required for all access to internal
tools, cloud consoles, and the production database. Access reviews are
conducted quarterly, and any account unused for 90 days is automatically
disabled.

## Data Classification

Data is classified into three tiers: Public, Internal, and Restricted.
Restricted data includes customer PII, payment information, and
authentication credentials. Restricted data must be encrypted at rest
(AES-256) and in transit (TLS 1.2 or higher). Internal data may be shared
within the company but not externally without approval from the data
owner.

## Incident Response

Any suspected security incident must be reported to the Security team
within one hour of discovery via the #security-incidents channel or the
security@company.example mailbox. The Security team follows a documented
runbook: containment, eradication, recovery, and a post-incident review
within five business days. Customers are notified within 72 hours if their
data was affected, in line with regulatory requirements.

## Password and Credential Requirements

Passwords must be at least 14 characters and are checked against a
breached-password list at creation time. Credentials must never be stored
in source code, configuration files committed to version control, or
shared over chat. All secrets are stored in the company's secrets manager
and rotated at least every 180 days.

## Third-Party Vendors

Any vendor with access to Restricted data must complete a security
questionnaire and sign a data processing agreement before onboarding.
Vendor access is reviewed annually alongside the standard access review.

## Enforcement

Violations of this policy may result in revoked system access and, for
repeated or severe violations, disciplinary action up to and including
termination. Employees who identify a policy gap are encouraged to report
it to Security rather than work around it.
