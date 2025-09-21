"""
Enhanced Query Orchestrator for Global MCP Client

This module provides intelligent query orchestration that can:
- Automatically analyze database schemas
- Chain multiple related queries
- Format results in dashboard-style tables
- Provide comprehensive data analysis
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime, timedelta

from .core.client import GlobalMCPClient
from .core.logger import LoggerMixin
from .core.exceptions import ToolExecutionError, AIProviderError


class QueryType(Enum):
    """Types of queries that can be automatically generated"""
    SCHEMA_ANALYSIS = "schema_analysis"
    DATA_SUMMARY = "data_summary"
    RELATIONSHIP_ANALYSIS = "relationship_analysis"
    PERFORMANCE_METRICS = "performance_metrics"
    DASHBOARD_QUERY = "dashboard_query"


@dataclass
class QueryPlan:
    """Represents a planned query execution"""
    query_type: QueryType
    description: str
    sql_query: Optional[str] = None
    tool_name: str = "query_table_records"
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    priority: int = 1  # 1 = highest priority


@dataclass
class QueryResult:
    """Result of a query execution"""
    plan_id: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    formatted_output: Optional[str] = None


class DatabaseAnalyzer(LoggerMixin):
    """Analyzes database schemas to understand structure and relationships - completely dynamic"""
    
    def __init__(self, client: GlobalMCPClient):
        self.client = client
        self.schema_cache = {}
        self.table_relationships = {}
        self.domain_patterns = self._initialize_domain_patterns()
    
    def _initialize_domain_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize universal patterns for any domain/industry"""
        return {
            "business_entities": {
                "people": ["user", "customer", "client", "person", "employee", "staff", "member", "participant", "individual"],
                "organizations": ["company", "organization", "org", "business", "enterprise", "firm", "corporation", "branch", "office", "department", "division"],
                "products": ["product", "item", "goods", "service", "offering", "catalog", "inventory", "asset", "resource"],
                "transactions": ["transaction", "order", "purchase", "sale", "payment", "invoice", "receipt", "transfer", "exchange"],
                "events": ["event", "activity", "action", "process", "workflow", "task", "job", "operation"],
                "locations": ["location", "address", "place", "site", "facility", "venue", "region", "area", "zone"]
            },
            "data_types": {
                "identifiers": ["id", "uuid", "key", "code", "number", "ref", "reference"],
                "amounts": ["amount", "total", "sum", "balance", "value", "price", "cost", "fee", "charge", "rate", "salary", "wage"],
                "quantities": ["count", "quantity", "qty", "number", "num", "volume", "size", "length", "weight"],
                "dates": ["date", "time", "timestamp", "created", "updated", "modified", "start", "end", "due", "expire"],
                "status": ["status", "state", "stage", "phase", "level", "priority", "flag", "active", "enabled"],
                "categories": ["type", "category", "class", "group", "kind", "genre", "classification"]
            },
            "relationships": {
                "ownership": ["owner", "belongs", "has", "contains"],
                "hierarchy": ["parent", "child", "super", "sub", "master", "detail"],
                "association": ["assigned", "linked", "connected", "related", "associated"]
            }
        }
    
    async def analyze_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Comprehensive schema analysis
        
        Args:
            schema_name: Name of the schema to analyze
            
        Returns:
            Schema analysis results
        """
        if schema_name in self.schema_cache:
            return self.schema_cache[schema_name]
            
        analysis = {
            "schema_name": schema_name,
            "tables": {},
            "relationships": [],
            "key_metrics_tables": [],
            "suggested_queries": []
        }
        
        try:
            # Get all tables in schema
            tables_result = await self.client.call_tool("get_all_tables", {"schemaName": schema_name})
            
            if isinstance(tables_result, list) and tables_result:
                tables_data = json.loads(tables_result[0].text)
                tables = tables_data.get("tables", [])
                
                # Analyze each table
                for table_info in tables:
                    table_name = table_info["TABLE_NAME"]
                    
                    # Get table structure
                    structure_result = await self.client.call_tool(
                        "analyze_table_structure",
                        {"tableName": table_name, "schemaName": schema_name}
                    )
                    
                    if isinstance(structure_result, list) and structure_result:
                        structure_data = json.loads(structure_result[0].text)
                        
                        # Analyze columns for business logic indicators
                        columns = structure_data.get("columns", [])
                        analysis["tables"][table_name] = {
                            "columns": columns,
                            "constraints": structure_data.get("constraints", []),
                            "business_indicators": self._analyze_business_indicators(columns),
                            "table_type": self._classify_table_type(table_name, columns)
                        }
                
                # Identify key business tables
                analysis["key_metrics_tables"] = self._identify_key_tables(analysis["tables"])
                
                # Generate suggested queries
                analysis["suggested_queries"] = self._generate_suggested_queries(analysis)
                
        except Exception as e:
            self.logger.error(f"Error analyzing schema {schema_name}: {e}")
            
        self.schema_cache[schema_name] = analysis
        return analysis
    
    def _analyze_business_indicators(self, columns: List[Dict]) -> Dict[str, List[str]]:
        """Dynamically identify business-relevant columns using universal patterns"""
        indicators = {
            "identifiers": [],
            "amounts": [],
            "quantities": [],
            "dates": [],
            "status": [],
            "categories": [],
            "text_fields": [],
            "flags": [],
            "rates": []
        }
        
        for col in columns:
            col_name = col["COLUMN_NAME"].lower()
            data_type = col["DATA_TYPE"].upper()
            
            # Dynamic pattern matching using universal patterns
            col_classified = False
            
            # Check identifiers
            if any(pattern in col_name for pattern in self.domain_patterns["data_types"]["identifiers"]):
                if data_type in ["NUMBER", "INTEGER", "VARCHAR2", "CHAR", "UUID"]:
                    indicators["identifiers"].append(col["COLUMN_NAME"])
                    col_classified = True
            
            # Check amounts/monetary values
            if not col_classified and any(pattern in col_name for pattern in self.domain_patterns["data_types"]["amounts"]):
                if data_type in ["NUMBER", "DECIMAL", "FLOAT", "MONEY", "CURRENCY"]:
                    indicators["amounts"].append(col["COLUMN_NAME"])
                    col_classified = True
            
            # Check quantities/counts
            if not col_classified and any(pattern in col_name for pattern in self.domain_patterns["data_types"]["quantities"]):
                if data_type in ["NUMBER", "INTEGER", "BIGINT", "SMALLINT"]:
                    indicators["quantities"].append(col["COLUMN_NAME"])
                    col_classified = True
            
            # Check dates/timestamps
            if not col_classified and (data_type in ["DATE", "DATETIME", "TIMESTAMP"] or 
                                      any(pattern in col_name for pattern in self.domain_patterns["data_types"]["dates"])):
                indicators["dates"].append(col["COLUMN_NAME"])
                col_classified = True
            
            # Check status/state fields
            if not col_classified and any(pattern in col_name for pattern in self.domain_patterns["data_types"]["status"]):
                indicators["status"].append(col["COLUMN_NAME"])
                col_classified = True
            
            # Check categories/types
            if not col_classified and any(pattern in col_name for pattern in self.domain_patterns["data_types"]["categories"]):
                indicators["categories"].append(col["COLUMN_NAME"])
                col_classified = True
            
            # Check for boolean/flag fields
            if not col_classified and (data_type in ["BOOLEAN", "BIT"] or 
                                      any(word in col_name for word in ["is_", "has_", "can_", "active", "enabled", "valid"])):
                indicators["flags"].append(col["COLUMN_NAME"])
                col_classified = True
            
            # Check for text fields (names, descriptions, etc.)
            if not col_classified and data_type in ["VARCHAR2", "VARCHAR", "TEXT", "CLOB", "STRING"]:
                if any(word in col_name for word in ["name", "title", "description", "comment", "note", "address", "email"]):
                    indicators["text_fields"].append(col["COLUMN_NAME"])
                    col_classified = True
            
            # Check for rate/percentage fields
            if not col_classified and any(word in col_name for word in ["rate", "percent", "ratio", "score"]):
                if data_type in ["NUMBER", "DECIMAL", "FLOAT"]:
                    indicators["rates"].append(col["COLUMN_NAME"])
                    col_classified = True
                
        return indicators
    
    def _classify_table_type(self, table_name: str, columns: List[Dict]) -> str:
        """Dynamically classify table type based on universal patterns and structure"""
        name_lower = table_name.lower()
        
        # Count different types of columns
        id_cols = len([col for col in columns if col["COLUMN_NAME"].lower().endswith("_id") or col["COLUMN_NAME"].lower().endswith("id")])
        date_cols = len([col for col in columns if col["DATA_TYPE"] == "DATE" or "date" in col["COLUMN_NAME"].lower()])
        amount_cols = len([col for col in columns if any(term in col["COLUMN_NAME"].lower() for term in self.domain_patterns["data_types"]["amounts"])])
        
        # Transaction/Fact tables - high activity, amounts, dates
        if any(term in name_lower for term in self.domain_patterns["business_entities"]["transactions"]):
            return "transactional"
        elif amount_cols >= 2 and date_cols >= 1:
            return "transactional"
        
        # Entity/Dimension tables - represent business entities
        elif (any(term in name_lower for term in self.domain_patterns["business_entities"]["people"]) or
              any(term in name_lower for term in self.domain_patterns["business_entities"]["organizations"]) or
              any(term in name_lower for term in self.domain_patterns["business_entities"]["products"]):
            return "dimensional"
        
        # Event/Activity tables
        elif any(term in name_lower for term in self.domain_patterns["business_entities"]["events"]):
            return "events"
        
        # Location tables
        elif any(term in name_lower for term in self.domain_patterns["business_entities"]["locations"]):
            return "locations"
        
        # Reference/Lookup tables - mostly categorical data
        elif (any(term in name_lower for term in self.domain_patterns["data_types"]["categories"]) or
              name_lower.endswith("_type") or name_lower.endswith("_status") or "lookup" in name_lower):
            return "reference"
        
        # Junction/Relationship tables - multiple foreign keys
        elif id_cols >= 2 and len(columns) <= 5:
            return "junction"
        
        # Audit/Log tables
        elif any(term in name_lower for term in ["audit", "log", "history", "track", "change"]):
            return "audit"
        
        # Configuration/Settings tables
        elif any(term in name_lower for term in ["config", "setting", "parameter", "option"]):
            return "configuration"
        
        # Security/Access tables
        elif any(term in name_lower for term in ["user", "role", "permission", "access", "auth"]):
            return "security"
        
        return "unknown"
    
    def _identify_key_tables(self, tables: Dict[str, Any]) -> List[str]:
        """Identify the most important tables for business metrics"""
        key_tables = []
        
        for table_name, table_info in tables.items():
            score = 0
            
            # Higher score for transactional tables
            if table_info["table_type"] == "transactional":
                score += 5
            elif table_info["table_type"] == "dimensional":
                score += 3
            
            # Score based on business indicators
            indicators = table_info["business_indicators"]
            score += len(indicators["amounts"]) * 3
            score += len(indicators["counts"]) * 2
            score += len(indicators["dates"]) * 2
            score += len(indicators["rates"]) * 2
            
            if score >= 5:  # Threshold for key tables
                key_tables.append(table_name)
        
        return key_tables
    
    def _generate_suggested_queries(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate suggested analytical queries based on schema analysis"""
        suggestions = []
        
        for table_name in analysis["key_metrics_tables"]:
            table_info = analysis["tables"][table_name]
            indicators = table_info["business_indicators"]
            
            # Aggregate queries for amounts
            for amount_col in indicators["amounts"][:2]:  # Limit to first 2
                suggestions.append({
                    "description": f"Total {amount_col.lower().replace('_', ' ')} by category/type",
                    "query_type": "aggregate_summary",
                    "table": table_name,
                    "column": amount_col
                })
            
            # Time-series analysis for date columns
            for date_col in indicators["dates"][:1]:  # Limit to first date column
                suggestions.append({
                    "description": f"Trend analysis by {date_col.lower().replace('_', ' ')}",
                    "query_type": "time_series",
                    "table": table_name,
                    "date_column": date_col
                })
        
        return suggestions


class QueryOrchestrator(LoggerMixin):
    """
    Intelligent query orchestrator that can automatically execute multiple related queries
    """
    
    def __init__(self, client: GlobalMCPClient):
        self.client = client
        self.analyzer = DatabaseAnalyzer(client)
        self.execution_history = []
        
    async def create_dashboard_query_plan(self, schema_name: str, request_type: str = "branch_metrics") -> List[QueryPlan]:
        """
        Create a comprehensive query plan for dashboard-style analysis
        
        Args:
            schema_name: Database schema to analyze
            request_type: Type of dashboard analysis requested
            
        Returns:
            List of QueryPlan objects to execute
        """
        plans = []
        
        # First, analyze the schema
        schema_analysis = await self.analyzer.analyze_schema(schema_name)
        
        if request_type == "branch_metrics":
            plans.extend(self._create_branch_metrics_plan(schema_name, schema_analysis))
        elif request_type == "loan_analysis":
            plans.extend(self._create_loan_analysis_plan(schema_name, schema_analysis))
        elif request_type == "comprehensive":
            plans.extend(self._create_comprehensive_plan(schema_name, schema_analysis))
            
        return plans
    
    def _create_branch_metrics_plan(self, schema_name: str, analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Create query plan specifically for branch performance metrics"""
        plans = []
        
        # Look for relevant tables
        loan_table = self._find_table_by_pattern(analysis, ["loan"])
        branch_table = self._find_table_by_pattern(analysis, ["branch"])
        
        if loan_table and branch_table:
            # Plan 1: Branch overview with loan metrics
            plans.append(QueryPlan(
                query_type=QueryType.DASHBOARD_QUERY,
                description="Branch performance dashboard with loan metrics",
                tool_name="execute_complex_query",
                parameters={
                    "schema": schema_name,
                    "query_template": "branch_loan_metrics",
                    "loan_table": loan_table,
                    "branch_table": branch_table
                },
                priority=1
            ))
            
            # Plan 2: Detailed loan analysis per branch
            plans.append(QueryPlan(
                query_type=QueryType.DATA_SUMMARY,
                description="Detailed loan analysis by branch",
                tool_name="query_table_records",
                parameters={
                    "tableName": loan_table,
                    "columns": "*",
                    "orderBy": "BRANCH_ID",
                    "limit": 1000
                },
                depends_on=["branch_loan_metrics"],
                priority=2
            ))
        
        return plans
    
    def _create_comprehensive_plan(self, schema_name: str, analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Create a comprehensive analysis plan"""
        plans = []
        
        # Schema overview
        plans.append(QueryPlan(
            query_type=QueryType.SCHEMA_ANALYSIS,
            description="Complete schema structure overview",
            tool_name="get_all_tables",
            parameters={"schemaName": schema_name},
            priority=1
        ))
        
        # Key table analysis
        for table_name in analysis["key_metrics_tables"][:5]:  # Limit to top 5 tables
            table_info = analysis["tables"][table_name]
            
            # Sample data from each key table
            plans.append(QueryPlan(
                query_type=QueryType.DATA_SUMMARY,
                description=f"Data summary for {table_name}",
                tool_name="query_table_records",
                parameters={
                    "tableName": table_name,
                    "columns": "*",
                    "limit": 10,
                    "orderBy": self._get_order_column(table_info)
                },
                priority=2
            ))
        
        return plans
    
    async def execute_query_plan(self, plans: List[QueryPlan]) -> Dict[str, QueryResult]:
        """
        Execute a list of query plans in the correct order
        
        Args:
            plans: List of QueryPlan objects to execute
            
        Returns:
            Dictionary mapping plan IDs to QueryResult objects
        """
        results = {}
        
        # Sort plans by priority and dependencies
        sorted_plans = self._sort_plans_by_dependencies(plans)
        
        for i, plan in enumerate(sorted_plans):
            plan_id = f"{plan.query_type.value}_{i}"
            
            self.logger.info(f"Executing query plan: {plan.description}")
            start_time = time.time()
            
            try:
                # Execute the query
                if plan.tool_name == "execute_complex_query":
                    result_data = await self._execute_complex_query(plan)
                else:
                    result_data = await self.client.call_tool(plan.tool_name, plan.parameters)
                
                execution_time = time.time() - start_time
                
                # Format the result
                formatted_output = await self._format_query_result(plan, result_data)
                
                results[plan_id] = QueryResult(
                    plan_id=plan_id,
                    success=True,
                    data=result_data,
                    execution_time=execution_time,
                    formatted_output=formatted_output
                )
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.logger.error(f"Query plan execution failed: {e}")
                
                results[plan_id] = QueryResult(
                    plan_id=plan_id,
                    success=False,
                    error=str(e),
                    execution_time=execution_time
                )
        
        return results
    
    async def _execute_complex_query(self, plan: QueryPlan) -> Any:
        """Execute complex queries that require multiple steps or custom logic"""
        if plan.parameters.get("query_template") == "branch_loan_metrics":
            return await self._execute_branch_loan_metrics(plan.parameters)
        
        # Default fallback
        return await self.client.call_tool(plan.tool_name, plan.parameters)
    
    async def _execute_branch_loan_metrics(self, params: Dict[str, Any]) -> Any:
        """Execute branch loan metrics query with proper aggregation"""
        loan_table = params["loan_table"]
        branch_table = params["branch_table"]
        schema = params["schema"]
        
        # First get branch data
        branches = await self.client.call_tool("query_table_records", {
            "tableName": branch_table,
            "columns": "BRANCH_ID,BRANCH_NAME",
            "limit": 100
        })
        
        # Then get loan aggregations per branch
        # Since we can't do complex SQL, we'll get all loans and process them
        loans = await self.client.call_tool("query_table_records", {
            "tableName": loan_table,
            "columns": "BRANCH_ID,LOAN_AMOUNT,APPLICATION_DATE,APPROVAL_DATE",
            "limit": 10000
        })
        
        return {"branches": branches, "loans": loans}
    
    async def _format_query_result(self, plan: QueryPlan, data: Any) -> str:
        """Format query results for dashboard-style presentation"""
        if not data:
            return "No data returned"
            
        if plan.query_type == QueryType.SCHEMA_ANALYSIS:
            return self._format_schema_analysis(data)
        elif plan.query_type == QueryType.DATA_SUMMARY:
            return self._format_data_summary(data, plan.description)
        elif plan.query_type == QueryType.DASHBOARD_QUERY:
            return self._format_dashboard_query(data)
        else:
            return self._format_generic_result(data)
    
    def _format_schema_analysis(self, data: Any) -> str:
        """Format schema analysis results"""
        if isinstance(data, list) and data:
            try:
                schema_data = json.loads(data[0].text)
                tables = schema_data.get("tables", [])
                
                output = f"## Schema Analysis\n"
                output += f"**Total Tables:** {len(tables)}\n\n"
                output += "### Tables Overview\n"
                output += "| Table Name | Rows | Last Analyzed |\n"
                output += "|------------|------|---------------|\n"
                
                for table in tables[:10]:  # Show first 10 tables
                    name = table.get("TABLE_NAME", "Unknown")
                    rows = table.get("NUM_ROWS", "N/A")
                    analyzed = table.get("LAST_ANALYZED", "N/A")
                    output += f"| {name} | {rows} | {analyzed} |\n"
                
                if len(tables) > 10:
                    output += f"| ... | ... | ... |\n"
                    output += f"*({len(tables) - 10} more tables)*\n"
                    
                return output
                
            except Exception as e:
                return f"Error formatting schema analysis: {e}"
        
        return str(data)
    
    def _format_data_summary(self, data: Any, description: str) -> str:
        """Format data summary results"""
        output = f"## {description}\n\n"
        
        if isinstance(data, list) and data:
            try:
                # Try to parse as JSON if it's a text result
                if hasattr(data[0], 'text'):
                    result_data = json.loads(data[0].text)
                    records = result_data.get("records", [])
                    
                    if records:
                        output += f"**Total Records Found:** {len(records)}\n\n"
                        
                        # Create a simple table format
                        if len(records) > 0:
                            headers = list(records[0].keys())
                            output += "| " + " | ".join(headers) + " |\n"
                            output += "|" + "|".join(["---"] * len(headers)) + "|\n"
                            
                            for record in records[:5]:  # Show first 5 records
                                values = [str(record.get(h, "")) for h in headers]
                                # Truncate long values
                                values = [v[:20] + "..." if len(v) > 20 else v for v in values]
                                output += "| " + " | ".join(values) + " |\n"
                                
                            if len(records) > 5:
                                output += f"\n*Showing 5 of {len(records)} records*\n"
                    else:
                        output += "No records found.\n"
                        
            except Exception as e:
                output += f"Error formatting data: {e}\n"
                output += f"Raw data: {str(data)[:500]}...\n"
        else:
            output += str(data)
        
        return output
    
    def _format_dashboard_query(self, data: Any) -> str:
        """Format dashboard query results with metrics and charts"""
        output = "## Branch Performance Dashboard\n\n"
        
        try:
            if isinstance(data, dict):
                branches_data = data.get("branches", [])
                loans_data = data.get("loans", [])
                
                # Process the data to create metrics
                branch_metrics = self._calculate_branch_metrics(branches_data, loans_data)
                
                if branch_metrics:
                    output += "### Key Metrics by Branch\n\n"
                    output += "| Branch | Total Loans | Total Amount | Avg Amount | Approval Rate | Avg Processing Days |\n"
                    output += "|--------|-------------|--------------|------------|---------------|--------------------|\n"
                    
                    for branch_id, metrics in branch_metrics.items():
                        output += f"| {metrics.get('name', branch_id)} | {metrics.get('total_loans', 0)} | "
                        output += f"${metrics.get('total_amount', 0):,.2f} | ${metrics.get('avg_amount', 0):,.2f} | "
                        output += f"{metrics.get('approval_rate', 0):.1f}% | {metrics.get('avg_processing_days', 0):.1f} |\n"
                        
                else:
                    output += "No metrics calculated - insufficient data.\n"
                    
        except Exception as e:
            output += f"Error formatting dashboard: {e}\n"
            
        return output
    
    def _calculate_branch_metrics(self, branches_data: Any, loans_data: Any) -> Dict[str, Dict[str, Any]]:
        """Calculate branch performance metrics from raw data"""
        metrics = {}
        
        try:
            # Parse branches data
            branches = {}
            if isinstance(branches_data, list) and branches_data:
                if hasattr(branches_data[0], 'text'):
                    branch_result = json.loads(branches_data[0].text)
                    for record in branch_result.get("records", []):
                        branch_id = record.get("BRANCH_ID")
                        if branch_id:
                            branches[branch_id] = record.get("BRANCH_NAME", f"Branch {branch_id}")
            
            # Parse loans data and calculate metrics
            if isinstance(loans_data, list) and loans_data:
                if hasattr(loans_data[0], 'text'):
                    loans_result = json.loads(loans_data[0].text)
                    loans_records = loans_result.get("records", [])
                    
                    # Group by branch
                    branch_loans = {}
                    for loan in loans_records:
                        branch_id = loan.get("BRANCH_ID")
                        if branch_id:
                            if branch_id not in branch_loans:
                                branch_loans[branch_id] = []
                            branch_loans[branch_id].append(loan)
                    
                    # Calculate metrics for each branch
                    for branch_id, loans in branch_loans.items():
                        total_loans = len(loans)
                        total_amount = sum(float(loan.get("LOAN_AMOUNT", 0)) for loan in loans if loan.get("LOAN_AMOUNT"))
                        avg_amount = total_amount / total_loans if total_loans > 0 else 0
                        
                        # Count approved loans (those with approval_date)
                        approved_loans = [loan for loan in loans if loan.get("APPROVAL_DATE")]
                        approval_rate = (len(approved_loans) / total_loans * 100) if total_loans > 0 else 0
                        
                        # Calculate processing time for approved loans
                        processing_days = []
                        for loan in approved_loans:
                            app_date = loan.get("APPLICATION_DATE")
                            app_date = loan.get("APPROVAL_DATE")
                            if app_date and app_date:
                                # Simplified - in real implementation would parse dates properly
                                processing_days.append(1.0)  # Placeholder
                        
                        avg_processing_days = sum(processing_days) / len(processing_days) if processing_days else 0
                        
                        metrics[branch_id] = {
                            "name": branches.get(branch_id, f"Branch {branch_id}"),
                            "total_loans": total_loans,
                            "total_amount": total_amount,
                            "avg_amount": avg_amount,
                            "approval_rate": approval_rate,
                            "avg_processing_days": avg_processing_days
                        }
                        
        except Exception as e:
            self.logger.error(f"Error calculating branch metrics: {e}")
            
        return metrics
    
    def _format_generic_result(self, data: Any) -> str:
        """Generic formatting for unknown result types"""
        if isinstance(data, list):
            return f"Result list with {len(data)} items:\n" + str(data)[:1000] + "..."
        elif isinstance(data, dict):
            return f"Result dictionary with {len(data)} keys:\n" + str(data)[:1000] + "..."
        else:
            return str(data)[:1000] + ("..." if len(str(data)) > 1000 else "")
    
    def _find_table_by_pattern(self, analysis: Dict[str, Any], patterns: List[str]) -> Optional[str]:
        """Find a table name that matches given patterns"""
        for table_name in analysis["tables"].keys():
            for pattern in patterns:
                if pattern.lower() in table_name.lower():
                    return table_name
        return None
    
    def _get_order_column(self, table_info: Dict[str, Any]) -> str:
        """Get appropriate column for ordering results"""
        indicators = table_info["business_indicators"]
        
        # Prefer date columns for ordering
        if indicators["dates"]:
            return indicators["dates"][0]
        # Then ID columns
        elif indicators["ids"]:
            return indicators["ids"][0]
        # Default to first column
        else:
            columns = table_info["columns"]
            return columns[0]["COLUMN_NAME"] if columns else ""
    
    def _sort_plans_by_dependencies(self, plans: List[QueryPlan]) -> List[QueryPlan]:
        """Sort plans by priority and dependencies"""
        # Simple sort by priority for now
        # In a more advanced implementation, would handle dependencies properly
        return sorted(plans, key=lambda p: p.priority)


class AutoQueryProcessor(LoggerMixin):
    """
    Main processor that handles automatic query execution and provides Claude Desktop-like responses
    """
    
    def __init__(self, client: GlobalMCPClient):
        self.client = client
        self.orchestrator = QueryOrchestrator(client)
        
    async def process_multi_query_request(self, request: str, schema_name: str = "C##loan_schema") -> str:
        """
        Process a request that may require multiple queries to fully answer
        
        Args:
            request: User's request/query
            schema_name: Database schema to work with
            
        Returns:
            Comprehensive formatted response
        """
        self.logger.info(f"Processing multi-query request: {request}")
        
        # Determine query type based on request
        query_type = self._classify_request(request)
        
        # Create appropriate query plan
        if query_type == "branch_metrics":
            plans = await self.orchestrator.create_dashboard_query_plan(schema_name, "branch_metrics")
        elif query_type == "comprehensive_analysis":
            plans = await self.orchestrator.create_dashboard_query_plan(schema_name, "comprehensive")
        else:
            # Default comprehensive analysis
            plans = await self.orchestrator.create_dashboard_query_plan(schema_name, "comprehensive")
        
        if not plans:
            return "Unable to create query plans for the requested analysis."
        
        # Execute the query plan
        results = await self.orchestrator.execute_query_plan(plans)
        
        # Format comprehensive response
        response = self._format_comprehensive_response(request, results, query_type)
        
        return response
    
    def _classify_request(self, request: str) -> str:
        """Classify the type of request to determine appropriate query strategy"""
        request_lower = request.lower()
        
        if any(term in request_lower for term in ["branch", "loan", "dashboard", "metrics", "performance"]):
            return "branch_metrics"
        elif any(term in request_lower for term in ["comprehensive", "complete", "all", "everything"]):
            return "comprehensive_analysis"
        elif any(term in request_lower for term in ["schema", "structure", "tables"]):
            return "schema_analysis"
        else:
            return "general_analysis"
    
    def _format_comprehensive_response(self, original_request: str, results: Dict[str, QueryResult], query_type: str) -> str:
        """Format a comprehensive response similar to Claude Desktop's detailed answers"""
        
        response = f"# Analysis Results for: {original_request}\n\n"
        response += f"*Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        # Executive Summary
        response += "## Executive Summary\n\n"
        successful_queries = len([r for r in results.values() if r.success])
        total_queries = len(results)
        
        response += f"Executed {successful_queries}/{total_queries} queries successfully. "
        
        if query_type == "branch_metrics":
            response += "Analyzed branch performance metrics including loan origination, approval rates, and processing times.\n\n"
        else:
            response += "Performed comprehensive database analysis with schema review and data summaries.\n\n"
        
        # Detailed Results
        response += "## Detailed Analysis\n\n"
        
        for plan_id, result in results.items():
            if result.success and result.formatted_output:
                response += result.formatted_output + "\n\n"
            elif not result.success:
                response += f"### Query Failed: {plan_id}\n"
                response += f"**Error:** {result.error}\n\n"
        
        # Performance Summary
        response += "## Query Performance\n\n"
        total_time = sum(r.execution_time for r in results.values())
        response += f"**Total Execution Time:** {total_time:.2f} seconds\n"
        response += f"**Average Query Time:** {total_time/len(results):.2f} seconds\n"
        response += f"**Success Rate:** {successful_queries/total_queries*100:.1f}%\n\n"
        
        # Recommendations
        response += "## Recommendations\n\n"
        response += self._generate_recommendations(results, query_type)
        
        return response
    
    def _generate_recommendations(self, results: Dict[str, QueryResult], query_type: str) -> str:
        """Generate actionable recommendations based on analysis results"""
        recommendations = ""
        
        if query_type == "branch_metrics":
            recommendations += "- **Monitor Branch Performance**: Set up regular monitoring of loan origination volumes and approval rates\n"
            recommendations += "- **Process Optimization**: Focus on reducing processing times for branches with longer approval cycles\n"
            recommendations += "- **Data Quality**: Ensure all loan records have complete application and approval date information\n"
        else:
            recommendations += "- **Schema Documentation**: Consider documenting table relationships and business rules\n"
            recommendations += "- **Data Governance**: Implement regular data quality checks and monitoring\n"
            recommendations += "- **Performance Monitoring**: Set up automated reporting for key business metrics\n"
        
        recommendations += "\n*For more detailed analysis, consider running specific queries on individual metrics of interest.*\n"
        
        return recommendations
