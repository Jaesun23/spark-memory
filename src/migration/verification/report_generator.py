"""
Report generation for Memory One Spark migration verification.

This module generates comprehensive reports from verification results
in multiple formats (JSON, HTML, Markdown, PDF).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum

from pydantic import BaseModel
import jinja2

from .data_verifier import DataVerificationReport
from .security_auditor import SecurityAuditReport, SecurityLevel
from .access_verifier import AccessVerificationReport
from ...utils.logging import get_logger

logger = get_logger(__name__)


class ReportFormat(str, Enum):
    """Supported report formats."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"
    SUMMARY = "summary"


class ReportConfig(BaseModel):
    """Configuration for report generation."""
    
    title: str = "Memory One Spark Migration Verification Report"
    organization: str = "Organization Name"
    include_timestamp: bool = True
    include_details: bool = True
    max_items_per_section: int = 100
    formats: List[ReportFormat] = [ReportFormat.HTML, ReportFormat.JSON]
    output_dir: Path = Path("./reports")


class CombinedReport(BaseModel):
    """Combined verification report from all verifiers."""
    
    report_id: str
    generation_time: datetime
    title: str
    organization: str
    
    # Individual reports
    data_verification: Optional[DataVerificationReport] = None
    security_audit: Optional[SecurityAuditReport] = None
    access_verification: Optional[AccessVerificationReport] = None
    
    # Overall summary
    overall_status: str = "UNKNOWN"
    executive_summary: Dict[str, Any] = {}
    key_findings: List[Dict[str, Any]] = []
    recommendations: List[str] = []


class ReportGenerator:
    """
    Generates comprehensive reports from verification results.
    
    Features:
    - Multiple output formats
    - Executive summaries
    - Detailed findings
    - Actionable recommendations
    - Visual charts and graphs (HTML)
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Report generation configuration
        """
        self.config = config or ReportConfig()
        self.combined_report: Optional[CombinedReport] = None
        
        # Setup Jinja2 for HTML templates
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath="./"),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    async def generate_report(
        self,
        data_report: Optional[DataVerificationReport] = None,
        security_report: Optional[SecurityAuditReport] = None,
        access_report: Optional[AccessVerificationReport] = None,
        formats: Optional[List[ReportFormat]] = None
    ) -> Dict[str, Path]:
        """
        Generate comprehensive report in specified formats.
        
        Args:
            data_report: Data verification results
            security_report: Security audit results
            access_report: Access verification results
            formats: Output formats (uses config default if not specified)
            
        Returns:
            Dictionary mapping format to output file path
        """
        # Create combined report
        report_id = f"verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.combined_report = CombinedReport(
            report_id=report_id,
            generation_time=datetime.now(),
            title=self.config.title,
            organization=self.config.organization,
            data_verification=data_report,
            security_audit=security_report,
            access_verification=access_report
        )
        
        # Generate overall summary
        self._generate_overall_summary()
        
        # Generate key findings
        self._extract_key_findings()
        
        # Generate recommendations
        self._consolidate_recommendations()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate reports in requested formats
        output_files = {}
        formats = formats or self.config.formats
        
        for format in formats:
            if format == ReportFormat.JSON:
                output_files[format] = await self._generate_json_report()
            elif format == ReportFormat.HTML:
                output_files[format] = await self._generate_html_report()
            elif format == ReportFormat.MARKDOWN:
                output_files[format] = await self._generate_markdown_report()
            elif format == ReportFormat.PDF:
                output_files[format] = await self._generate_pdf_report()
            elif format == ReportFormat.SUMMARY:
                output_files[format] = await self._generate_summary_report()
        
        logger.info(f"Generated {len(output_files)} report(s) for {report_id}")
        return output_files
    
    def _generate_overall_summary(self):
        """Generate overall summary from all reports."""
        summary = {
            "total_checks": 0,
            "passed_checks": 0,
            "failed_checks": 0,
            "critical_issues": 0,
            "warnings": 0
        }
        
        # Data verification summary
        if self.combined_report.data_verification:
            dv = self.combined_report.data_verification
            summary["total_checks"] += dv.total_items
            summary["passed_checks"] += dv.verified_items
            summary["failed_checks"] += dv.failed_items
            summary["data_integrity_rate"] = (
                dv.verified_items / dv.total_items * 100
            ) if dv.total_items > 0 else 0
        
        # Security audit summary
        if self.combined_report.security_audit:
            sa = self.combined_report.security_audit
            summary["total_checks"] += len(sa.findings)
            summary["critical_issues"] += sa.summary.get("critical", 0)
            summary["critical_issues"] += sa.summary.get("high", 0)
            summary["warnings"] += sa.summary.get("medium", 0)
            summary["warnings"] += sa.summary.get("low", 0)
            summary["security_findings"] = len(sa.findings)
        
        # Access verification summary
        if self.combined_report.access_verification:
            av = self.combined_report.access_verification
            summary["total_checks"] += av.access_rules_verified
            summary["access_violations"] = len(av.violations_found)
            summary["unauthorized_users"] = len(av.unauthorized_users)
            summary["orphaned_resources"] = len(av.orphaned_resources)
        
        # Determine overall status
        if summary["critical_issues"] > 0 or summary["failed_checks"] > 0:
            self.combined_report.overall_status = "FAILED"
        elif summary["warnings"] > 0:
            self.combined_report.overall_status = "PASSED_WITH_WARNINGS"
        else:
            self.combined_report.overall_status = "PASSED"
        
        self.combined_report.executive_summary = summary
    
    def _extract_key_findings(self):
        """Extract key findings from all reports."""
        findings = []
        
        # Data verification findings
        if self.combined_report.data_verification:
            dv = self.combined_report.data_verification
            if dv.failed_items > 0:
                findings.append({
                    "severity": "high",
                    "category": "Data Integrity",
                    "finding": f"{dv.failed_items} items failed data verification",
                    "impact": "Data loss or corruption during migration"
                })
        
        # Security findings
        if self.combined_report.security_audit:
            sa = self.combined_report.security_audit
            for finding in sa.findings:
                if finding.level in [SecurityLevel.CRITICAL, SecurityLevel.HIGH]:
                    findings.append({
                        "severity": finding.level.value,
                        "category": finding.category,
                        "finding": finding.title,
                        "impact": finding.description
                    })
        
        # Access violations
        if self.combined_report.access_verification:
            av = self.combined_report.access_verification
            if av.violations_found:
                findings.append({
                    "severity": "high",
                    "category": "Access Control",
                    "finding": f"{len(av.violations_found)} access violations detected",
                    "impact": "Unauthorized access or permission misconfigurations"
                })
        
        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        findings.sort(key=lambda x: severity_order.get(x["severity"], 999))
        
        # Limit to top findings
        self.combined_report.key_findings = findings[:10]
    
    def _consolidate_recommendations(self):
        """Consolidate recommendations from all reports."""
        recommendations = []
        
        # Add critical recommendations first
        if self.combined_report.overall_status == "FAILED":
            recommendations.append(
                "CRITICAL: Address all failed verification checks before proceeding with migration"
            )
        
        # Security recommendations
        if self.combined_report.security_audit:
            recommendations.extend(self.combined_report.security_audit.recommendations)
        
        # Data integrity recommendations
        if self.combined_report.data_verification:
            dv = self.combined_report.data_verification
            if dv.failed_items > 0:
                recommendations.append(
                    f"Re-migrate {dv.failed_items} failed items with enhanced error handling"
                )
        
        # Access control recommendations
        if self.combined_report.access_verification:
            av = self.combined_report.access_verification
            if av.unauthorized_users:
                recommendations.append(
                    f"Remove or disable {len(av.unauthorized_users)} unauthorized users"
                )
            if av.orphaned_resources:
                recommendations.append(
                    f"Define access policies for {len(av.orphaned_resources)} orphaned resources"
                )
        
        self.combined_report.recommendations = recommendations[:15]  # Limit recommendations
    
    async def _generate_json_report(self) -> Path:
        """Generate JSON format report."""
        output_path = self.config.output_dir / f"{self.combined_report.report_id}.json"
        
        # Convert to dict and write
        report_dict = self.combined_report.dict()
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Generated JSON report: {output_path}")
        return output_path
    
    async def _generate_html_report(self) -> Path:
        """Generate HTML format report."""
        output_path = self.config.output_dir / f"{self.combined_report.report_id}.html"
        
        # HTML template
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ report.title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f0f0f0; padding: 20px; border-radius: 5px; }
        .status-passed { color: green; font-weight: bold; }
        .status-failed { color: red; font-weight: bold; }
        .status-warning { color: orange; font-weight: bold; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
        .finding { margin: 10px 0; padding: 10px; background: #f9f9f9; }
        .critical { border-left: 5px solid red; }
        .high { border-left: 5px solid orange; }
        .medium { border-left: 5px solid yellow; }
        .low { border-left: 5px solid green; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f0f0f0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ report.title }}</h1>
        <p>Organization: {{ report.organization }}</p>
        <p>Generated: {{ report.generation_time }}</p>
        <p>Overall Status: <span class="status-{{ report.overall_status.lower() }}">{{ report.overall_status }}</span></p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {% for key, value in report.executive_summary.items() %}
            <tr><td>{{ key.replace('_', ' ').title() }}</td><td>{{ value }}</td></tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Key Findings</h2>
        {% for finding in report.key_findings %}
        <div class="finding {{ finding.severity }}">
            <h3>{{ finding.finding }}</h3>
            <p><strong>Category:</strong> {{ finding.category }}</p>
            <p><strong>Impact:</strong> {{ finding.impact }}</p>
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Recommendations</h2>
        <ol>
        {% for rec in report.recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ol>
    </div>
    
    {% if report.data_verification %}
    <div class="section">
        <h2>Data Verification Details</h2>
        <p>Total Items: {{ report.data_verification.total_items }}</p>
        <p>Verified: {{ report.data_verification.verified_items }}</p>
        <p>Failed: {{ report.data_verification.failed_items }}</p>
    </div>
    {% endif %}
    
    {% if report.security_audit %}
    <div class="section">
        <h2>Security Audit Details</h2>
        <p>Total Findings: {{ report.security_audit.findings|length }}</p>
        <p>Critical/High: {{ report.security_audit.summary.critical + report.security_audit.summary.high }}</p>
        <p>Medium/Low: {{ report.security_audit.summary.medium + report.security_audit.summary.low }}</p>
    </div>
    {% endif %}
    
    {% if report.access_verification %}
    <div class="section">
        <h2>Access Verification Details</h2>
        <p>Rules Verified: {{ report.access_verification.access_rules_verified }}</p>
        <p>Violations Found: {{ report.access_verification.violations_found|length }}</p>
        <p>Unauthorized Users: {{ report.access_verification.unauthorized_users|length }}</p>
    </div>
    {% endif %}
</body>
</html>
        """
        
        # Render template
        template = self.jinja_env.from_string(template_str)
        html_content = template.render(report=self.combined_report)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {output_path}")
        return output_path
    
    async def _generate_markdown_report(self) -> Path:
        """Generate Markdown format report."""
        output_path = self.config.output_dir / f"{self.combined_report.report_id}.md"
        
        md_lines = [
            f"# {self.combined_report.title}",
            f"",
            f"**Organization:** {self.combined_report.organization}",
            f"**Generated:** {self.combined_report.generation_time}",
            f"**Overall Status:** {self.combined_report.overall_status}",
            f"",
            f"## Executive Summary",
            f""
        ]
        
        # Summary table
        md_lines.append("| Metric | Value |")
        md_lines.append("|--------|-------|")
        for key, value in self.combined_report.executive_summary.items():
            md_lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
        
        # Key findings
        md_lines.extend([
            "",
            "## Key Findings",
            ""
        ])
        
        for i, finding in enumerate(self.combined_report.key_findings, 1):
            md_lines.extend([
                f"### {i}. {finding['finding']}",
                f"- **Severity:** {finding['severity']}",
                f"- **Category:** {finding['category']}",
                f"- **Impact:** {finding['impact']}",
                ""
            ])
        
        # Recommendations
        md_lines.extend([
            "## Recommendations",
            ""
        ])
        
        for i, rec in enumerate(self.combined_report.recommendations, 1):
            md_lines.append(f"{i}. {rec}")
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(md_lines))
        
        logger.info(f"Generated Markdown report: {output_path}")
        return output_path
    
    async def _generate_pdf_report(self) -> Path:
        """Generate PDF format report."""
        # This would require additional dependencies like reportlab or weasyprint
        # For now, we'll create a placeholder
        output_path = self.config.output_dir / f"{self.combined_report.report_id}.pdf"
        
        # In a real implementation, you would:
        # 1. Generate HTML first
        # 2. Convert HTML to PDF using weasyprint or similar
        
        logger.warning("PDF generation not implemented - would require additional dependencies")
        return output_path
    
    async def _generate_summary_report(self) -> Path:
        """Generate brief summary report."""
        output_path = self.config.output_dir / f"{self.combined_report.report_id}_summary.txt"
        
        summary_lines = [
            f"VERIFICATION SUMMARY - {self.combined_report.report_id}",
            "=" * 50,
            f"Status: {self.combined_report.overall_status}",
            f"Generated: {self.combined_report.generation_time}",
            "",
            "KEY METRICS:",
        ]
        
        for key, value in self.combined_report.executive_summary.items():
            summary_lines.append(f"  {key}: {value}")
        
        summary_lines.extend([
            "",
            "TOP ISSUES:",
        ])
        
        for finding in self.combined_report.key_findings[:5]:
            summary_lines.append(f"  - [{finding['severity'].upper()}] {finding['finding']}")
        
        summary_lines.extend([
            "",
            "IMMEDIATE ACTIONS REQUIRED:",
        ])
        
        for rec in self.combined_report.recommendations[:5]:
            summary_lines.append(f"  - {rec}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(f"Generated summary report: {output_path}")
        return output_path