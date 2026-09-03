# Privacy Policy

This document covers the personal automation tools in this repository that
integrate with Google APIs (Gmail).

## What this is

This is a personal research project. The Google OAuth application registered
under this project ("claude desktop") is used exclusively by the project
owner (a single individual) to automate their own Gmail account for their own
research workflow — sending status emails and, optionally, reading their own
inbox via an MCP (Model Context Protocol) integration running locally.

## Who uses this

Only the developer/owner of this repository. This application is not
distributed to, or used by, any other end user.

## What data is accessed

- Gmail messages, labels, and settings in the owner's own Gmail account,
  accessed via the scopes `gmail.modify` and `gmail.settings.basic`.
- No data belonging to any other Google account is accessed.

## How the data is used

- Sending research-status emails (`test/scripts/send_email.py`) from the
  owner's own account to the owner's own designated recipient(s).
- Optionally, reading/searching the owner's own inbox through a local MCP
  server (`@gongrzhe/server-gmail-autoauth-mcp`) running on the owner's own
  machine, invoked interactively by the owner.

## Data storage and sharing

- OAuth credentials are stored locally on the owner's own server/machine
  (`~/.gmail-mcp/credentials.json`), never committed to this repository, and
  never transmitted to any third party.
- No data accessed through these scopes is shared with, sold to, or
  processed by any third party or external service. All processing happens
  locally on infrastructure controlled by the owner.

## Contact

For questions about this policy, contact the repository owner at the email
listed in the Google Cloud OAuth consent screen for this project.
