//! Audit logging for security events in the swarm inference system.
//!
//! This module provides comprehensive audit logging for security-related events
//! including authentication, authorization, access control, and system security
//! operations. All audit logs are structured and include relevant context.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::{error, info, warn};

/// Types of audit events that can occur in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Authentication events (login, logout, token validation)
    Authentication,
    /// Authorization events (permission checks, access control)
    Authorization,
    /// Access control events (resource access, data transmission)
    AccessControl,
    /// Node registration and management events
    NodeManagement,
    /// Model and data access events
    DataAccess,
    /// System security events (certificate validation, TLS issues)
    SystemSecurity,
    /// Administrative actions
    Administration,
}

/// Severity level for audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Comprehensive audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Unique event ID
    pub event_id: String,
    /// Timestamp in Unix epoch seconds
    pub timestamp: u64,
    /// Type of audit event
    pub event_type: AuditEventType,
    /// Severity level
    pub severity: AuditSeverity,
    /// Source of the event (IP address, node ID, etc.)
    pub source: String,
    /// User or entity performing the action (if applicable)
    pub user: Option<String>,
    /// Target of the action (resource, node, etc.)
    pub target: Option<String>,
    /// Action that was performed
    pub action: String,
    /// Result of the action (success/failure)
    pub result: AuditResult,
    /// Additional context and metadata
    pub context: serde_json::Value,
    /// Session ID if applicable
    pub session_id: Option<String>,
}

/// Result of an audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failure { reason: String },
    Denied { reason: String },
    Error { error: String },
}

/// Parameters for logging an audit event
pub struct AuditEventParams<'a> {
    pub event_type: AuditEventType,
    pub severity: AuditSeverity,
    pub source: &'a str,
    pub user: Option<&'a str>,
    pub target: Option<&'a str>,
    pub action: &'a str,
    pub result: AuditResult,
    pub context: serde_json::Value,
    pub session_id: Option<&'a str>,
}

/// Audit logger for recording security events
#[derive(Debug)]
pub struct AuditLogger {
    node_id: String,
    system_id: String,
}

impl AuditLogger {
    /// Create a new audit logger
    pub fn new(node_id: String, system_id: String) -> Self {
        Self { node_id, system_id }
    }

    /// Log an authentication event
    pub fn log_authentication(
        &self,
        source: &str,
        user: Option<&str>,
        action: &str,
        result: AuditResult,
        context: serde_json::Value,
    ) {
        self.log_event(AuditEventParams {
            event_type: AuditEventType::Authentication,
            severity: AuditSeverity::Medium,
            source,
            user,
            target: None,
            action,
            result,
            context,
            session_id: None,
        });
    }

    /// Log an authorization event
    pub fn log_authorization(
        &self,
        source: &str,
        user: Option<&str>,
        target: Option<&str>,
        action: &str,
        result: AuditResult,
        context: serde_json::Value,
    ) {
        let severity = match &result {
            AuditResult::Denied { .. } => AuditSeverity::High,
            _ => AuditSeverity::Medium,
        };

        self.log_event(AuditEventParams {
            event_type: AuditEventType::Authorization,
            severity,
            source,
            user,
            target,
            action,
            result,
            context,
            session_id: None,
        });
    }

    /// Log an access control event
    pub fn log_access_control(
        &self,
        source: &str,
        user: Option<&str>,
        target: &str,
        action: &str,
        result: AuditResult,
        context: serde_json::Value,
    ) {
        let severity = match &result {
            AuditResult::Denied { .. } => AuditSeverity::High,
            _ => AuditSeverity::Low,
        };

        self.log_event(AuditEventParams {
            event_type: AuditEventType::AccessControl,
            severity,
            source,
            user,
            target: Some(target),
            action,
            result,
            context,
            session_id: None,
        });
    }

    /// Log a node management event
    pub fn log_node_management(
        &self,
        source: &str,
        target: &str,
        action: &str,
        result: AuditResult,
        context: serde_json::Value,
    ) {
        self.log_event(AuditEventParams {
            event_type: AuditEventType::NodeManagement,
            severity: AuditSeverity::Medium,
            source,
            user: None,
            target: Some(target),
            action,
            result,
            context,
            session_id: None,
        });
    }

    /// Log a data access event
    pub fn log_data_access(
        &self,
        source: &str,
        user: Option<&str>,
        target: &str,
        action: &str,
        result: AuditResult,
        context: serde_json::Value,
    ) {
        let severity = match &result {
            AuditResult::Denied { .. } => AuditSeverity::Critical,
            _ => AuditSeverity::Medium,
        };

        self.log_event(AuditEventParams {
            event_type: AuditEventType::DataAccess,
            severity,
            source,
            user,
            target: Some(target),
            action,
            result,
            context,
            session_id: None,
        });
    }

    /// Log a system security event
    pub fn log_system_security(
        &self,
        source: &str,
        action: &str,
        result: AuditResult,
        context: serde_json::Value,
    ) {
        let severity = match &result {
            AuditResult::Failure { .. } | AuditResult::Error { .. } => AuditSeverity::High,
            _ => AuditSeverity::Medium,
        };

        self.log_event(AuditEventParams {
            event_type: AuditEventType::SystemSecurity,
            severity,
            source,
            user: None,
            target: None,
            action,
            result,
            context,
            session_id: None,
        });
    }

    /// Log an administrative action
    pub fn log_administration(
        &self,
        source: &str,
        user: &str,
        action: &str,
        result: AuditResult,
        context: serde_json::Value,
    ) {
        self.log_event(AuditEventParams {
            event_type: AuditEventType::Administration,
            severity: AuditSeverity::High,
            source,
            user: Some(user),
            target: None,
            action,
            result,
            context,
            session_id: None,
        });
    }

    /// Internal method to create and log an audit event
    fn log_event(&self, entry_params: AuditEventParams) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let event_id = format!("{}-{}-{}", self.system_id, self.node_id, timestamp);

        let entry = AuditLogEntry {
            event_id,
            timestamp,
            event_type: entry_params.event_type,
            severity: entry_params.severity,
            source: entry_params.source.to_string(),
            user: entry_params.user.map(|s| s.to_string()),
            target: entry_params.target.map(|s| s.to_string()),
            action: entry_params.action.to_string(),
            result: entry_params.result,
            context: entry_params.context,
            session_id: entry_params.session_id.map(|s| s.to_string()),
        };

        // Log to structured logging system
        self.log_to_tracing(&entry);

        // In a production system, you would also:
        // - Write to audit log files
        // - Send to centralized logging system
        // - Store in audit database
        // - Trigger alerts for critical events
    }

    /// Log the audit entry to the tracing system
    fn log_to_tracing(&self, entry: &AuditLogEntry) {
        let json_entry = serde_json::to_string(entry).unwrap_or_else(|_| "{}".to_string());

        match entry.severity {
            AuditSeverity::Critical => {
                error!(
                    audit_event = %json_entry,
                    event_type = ?entry.event_type,
                    severity = ?entry.severity,
                    source = %entry.source,
                    action = %entry.action,
                    "CRITICAL AUDIT EVENT"
                );
            }
            AuditSeverity::High => {
                error!(
                    audit_event = %json_entry,
                    event_type = ?entry.event_type,
                    severity = ?entry.severity,
                    source = %entry.source,
                    action = %entry.action,
                    "HIGH AUDIT EVENT"
                );
            }
            AuditSeverity::Medium => {
                warn!(
                    audit_event = %json_entry,
                    event_type = ?entry.event_type,
                    severity = ?entry.severity,
                    source = %entry.source,
                    action = %entry.action,
                    "MEDIUM AUDIT EVENT"
                );
            }
            AuditSeverity::Low => {
                info!(
                    audit_event = %json_entry,
                    event_type = ?entry.event_type,
                    severity = ?entry.severity,
                    source = %entry.source,
                    action = %entry.action,
                    "LOW AUDIT EVENT"
                );
            }
        }
    }
}

/// Global audit logger instance
use std::sync::OnceLock;
static AUDIT_LOGGER: OnceLock<AuditLogger> = OnceLock::new();

/// Initialize the global audit logger
pub fn init_audit_logger(node_id: String, system_id: String) {
    let logger = AuditLogger::new(node_id, system_id);
    AUDIT_LOGGER
        .set(logger)
        .expect("Audit logger already initialized");
}

/// Get the global audit logger instance
pub fn get_audit_logger() -> &'static AuditLogger {
    AUDIT_LOGGER.get().expect("Audit logger not initialized")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_logger_creation() {
        let logger = AuditLogger::new("node-1".to_string(), "system-1".to_string());
        assert_eq!(logger.node_id, "node-1");
        assert_eq!(logger.system_id, "system-1");
    }

    #[test]
    fn test_audit_log_entry_serialization() {
        let entry = AuditLogEntry {
            event_id: "test-event".to_string(),
            timestamp: 1234567890,
            event_type: AuditEventType::Authentication,
            severity: AuditSeverity::Medium,
            source: "127.0.0.1:8080".to_string(),
            user: Some("user1".to_string()),
            target: None,
            action: "login".to_string(),
            result: AuditResult::Success,
            context: serde_json::json!({"method": "password"}),
            session_id: Some("session-123".to_string()),
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("test-event"));
        assert!(json.contains("Authentication"));
    }

    #[test]
    fn test_global_audit_logger() {
        // Initialize the global logger
        init_audit_logger("test-node".to_string(), "test-system".to_string());

        // Get the logger
        let logger = get_audit_logger();
        assert_eq!(logger.node_id, "test-node");
        assert_eq!(logger.system_id, "test-system");
    }
}
