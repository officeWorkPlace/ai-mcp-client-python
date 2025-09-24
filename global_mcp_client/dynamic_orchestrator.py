"""
Universal Dynamic Query Orchestrator

This orchestrator is completely schema-agnostic and can work with ANY type of business data
by using pattern recognition, statistical analysis, and dynamic query generation.
It doesn't rely on any hardcoded business terms or domain-specific knowledge.
"""

import asyncio
import json
import time
import re
import statistics
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import Counter, defaultdict

from .core.client import GlobalMCPClient
from .core.logger import LoggerMixin
from .core.exceptions import ToolExecutionError, AIProviderError


class ColumnType(Enum):
    """Universal column types detected through pattern analysis"""
    IDENTIFIER = "identifier"
    NUMERIC_MEASURE = "numeric_measure" 
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT_CONTENT = "text_content"
    BOOLEAN_FLAG = "boolean_flag"
    FOREIGN_KEY = "foreign_key"
    UNKNOWN = "unknown"


class TableRole(Enum):
    """Universal table roles detected through structural analysis"""
    FACT = "fact"  # High-activity tables with measures
    DIMENSION = "dimension"  # Entity tables with attributes
    LOOKUP = "lookup"  # Reference/lookup tables
    BRIDGE = "bridge"  # Junction/many-to-many tables
    LOG = "log"  # Event/audit tables
    CONFIG = "config"  # Configuration tables
    UNKNOWN = "unknown"


@dataclass
class ColumnProfile:
    """Statistical profile of a column"""
    name: str
    data_type: str
    column_type: ColumnType
    nullable: bool
    unique_ratio: float = 0.0  # Ratio of unique values
    null_ratio: float = 0.0    # Ratio of null values
    text_length_stats: Dict[str, float] = field(default_factory=dict)
    numeric_stats: Dict[str, float] = field(default_factory=dict)
    sample_values: List[str] = field(default_factory=list)
    pattern_score: Dict[str, float] = field(default_factory=dict)


@dataclass 
class TableProfile:
    """Complete profile of a table"""
    name: str
    schema_name: str
    table_role: TableRole
    row_count: int
    columns: List[ColumnProfile]
    relationships: List[str] = field(default_factory=list)
    business_score: float = 0.0  # How important this table seems
    complexity_score: float = 0.0  # How complex this table is


@dataclass
class QueryPlan:
    """Dynamic query execution plan"""
    id: str
    description: str
    priority: int
    tool_name: str
    parameters: Dict[str, Any]
    expected_insight: str
    depends_on: List[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Result of running an analysis"""
    plan_id: str
    success: bool
    execution_time: float
    data: Any = None
    formatted_output: str = ""
    error: Optional[str] = None
    insights: List[str] = field(default_factory=list)


class UniversalSchemaAnalyzer(LoggerMixin):
    """
    Universal analyzer that can understand any schema through pattern recognition
    and statistical analysis, without domain-specific knowledge
    """
    
    def __init__(self, client: GlobalMCPClient):
        self.client = client
        self.analysis_cache = {}
        
    async def analyze_schema(self, schema_name: str) -> Dict[str, Any]:
        """
        Perform comprehensive, domain-agnostic schema analysis
        """
        if schema_name in self.analysis_cache:
            return self.analysis_cache[schema_name]
            
        self.logger.info(f"Starting universal analysis of schema: {schema_name}")
        
        analysis = {
            "schema_name": schema_name,
            "table_profiles": [],
            "relationships": [],
            "key_tables": [],
            "suggested_analyses": [],
            "domain_insights": []
        }
        
        try:
            # Step 1: Get all tables
            tables_result = await self.client.call_tool("get_all_tables", {"schemaName": schema_name})
            
            if isinstance(tables_result, list) and tables_result:
                tables_data = json.loads(tables_result[0].text)
                tables = tables_data.get("tables", [])
                
                # Step 2: Profile each table
                table_profiles = []
                for table_info in tables:
                    table_name = table_info["TABLE_NAME"]
                    profile = await self._profile_table(schema_name, table_name, table_info)
                    if profile:
                        table_profiles.append(profile)
                
                analysis["table_profiles"] = table_profiles
                
                # Step 3: Detect relationships
                analysis["relationships"] = self._detect_relationships(table_profiles)
                
                # Step 4: Identify key tables for analysis
                analysis["key_tables"] = self._identify_key_tables(table_profiles)
                
                # Step 5: Generate analysis suggestions
                analysis["suggested_analyses"] = self._generate_universal_analyses(table_profiles)
                
                # Step 6: Infer domain insights
                analysis["domain_insights"] = self._infer_domain_characteristics(table_profiles)

                # Step 7: Enhanced Business Intelligence Analysis
                domain_classification = self._classify_business_domain(table_profiles)
                analysis["business_domain"] = domain_classification
                analysis["business_insights"] = self._generate_business_insights(table_profiles, domain_classification)
                analysis["kpi_recommendations"] = self._generate_advanced_kpi_recommendations(table_profiles, domain_classification)
                
        except Exception as e:
            self.logger.error(f"Error in universal schema analysis: {e}")
            
        self.analysis_cache[schema_name] = analysis
        return analysis
    
    async def _profile_table(self, schema_name: str, table_name: str, table_info: Dict) -> Optional[TableProfile]:
        """Profile a single table comprehensively"""
        try:
            # Get detailed table structure
            structure_result = await self.client.call_tool(
                "analyze_table_structure",
                {"tableName": table_name, "schemaName": schema_name}
            )
            
            if not (isinstance(structure_result, list) and structure_result):
                return None
                
            structure_data = json.loads(structure_result[0].text)
            columns_data = structure_data.get("columns", [])
            
            # Sample some data for statistical analysis
            sample_result = await self.client.call_tool(
                "query_table_records",
                {"tableName": table_name, "columns": "*", "limit": 50}
            )
            
            sample_data = []
            if isinstance(sample_result, list) and sample_result:
                try:
                    sample_json = json.loads(sample_result[0].text)
                    sample_data = sample_json.get("records", [])
                except:
                    pass
            
            # Profile each column
            column_profiles = []
            for col_data in columns_data:
                profile = await self._profile_column(col_data, sample_data)
                column_profiles.append(profile)
            
            # Determine table role
            table_role = self._classify_table_role(table_name, column_profiles, sample_data)
            
            # Calculate business importance score
            business_score = self._calculate_business_score(table_name, column_profiles, sample_data)
            
            return TableProfile(
                name=table_name,
                schema_name=schema_name,
                table_role=table_role,
                row_count=table_info.get("NUM_ROWS", 0) or len(sample_data),
                columns=column_profiles,
                business_score=business_score,
                complexity_score=len(column_profiles) * 0.1 + len(sample_data) * 0.01
            )
            
        except Exception as e:
            self.logger.error(f"Error profiling table {table_name}: {e}")
            return None
    
    async def _profile_column(self, col_data: Dict, sample_data: List[Dict]) -> ColumnProfile:
        """Profile a single column using statistical analysis"""
        col_name = col_data["COLUMN_NAME"]
        data_type = col_data["DATA_TYPE"]
        nullable = col_data.get("NULLABLE", "Y") == "Y"
        
        # Extract sample values for this column
        sample_values = []
        for row in sample_data:
            val = row.get(col_name)
            if val is not None:
                sample_values.append(str(val))
        
        # Statistical analysis
        unique_ratio = len(set(sample_values)) / max(len(sample_values), 1) if sample_values else 0
        null_ratio = (len(sample_data) - len(sample_values)) / max(len(sample_data), 1)
        
        # Detect column type through pattern analysis
        column_type = self._detect_column_type(col_name, data_type, sample_values, unique_ratio)
        
        # Calculate pattern scores
        pattern_score = self._calculate_pattern_scores(col_name, sample_values)
        
        return ColumnProfile(
            name=col_name,
            data_type=data_type,
            column_type=column_type,
            nullable=nullable,
            unique_ratio=unique_ratio,
            null_ratio=null_ratio,
            sample_values=sample_values[:10],  # Keep first 10 samples
            pattern_score=pattern_score
        )
    
    def _detect_column_type(self, col_name: str, data_type: str, sample_values: List[str], unique_ratio: float) -> ColumnType:
        """Detect column type through universal pattern analysis"""
        name_lower = col_name.lower()
        
        # Identifier patterns - high uniqueness
        if unique_ratio > 0.9 or name_lower.endswith('id') or 'key' in name_lower:
            if any(fk_pattern in name_lower for fk_pattern in ['_id', 'ref', 'foreign']):
                return ColumnType.FOREIGN_KEY
            return ColumnType.IDENTIFIER
        
        # Temporal patterns
        if (data_type in ['DATE', 'DATETIME', 'TIMESTAMP'] or 
            any(temporal in name_lower for temporal in ['date', 'time', 'created', 'updated', 'start', 'end'])):
            return ColumnType.TEMPORAL
        
        # Boolean patterns
        if (data_type in ['BOOLEAN', 'BIT'] or 
            unique_ratio < 0.1 or  # Very few unique values
            any(bool_pattern in name_lower for bool_pattern in ['is_', 'has_', 'active', 'enabled', 'flag'])):
            return ColumnType.BOOLEAN_FLAG
        
        # Numeric measures - numeric with medium-high variance
        if data_type in ['NUMBER', 'DECIMAL', 'FLOAT', 'INTEGER']:
            if self._is_numeric_measure(sample_values):
                return ColumnType.NUMERIC_MEASURE
        
        # Categorical - low-medium uniqueness with text
        if unique_ratio < 0.5 and len(sample_values) > 0:
            return ColumnType.CATEGORICAL
        
        # Text content - high uniqueness with longer strings
        if data_type in ['VARCHAR2', 'VARCHAR', 'TEXT', 'CLOB']:
            avg_length = sum(len(v) for v in sample_values) / max(len(sample_values), 1)
            if avg_length > 20:  # Longer text suggests content
                return ColumnType.TEXT_CONTENT
        
        return ColumnType.UNKNOWN
    
    def _is_numeric_measure(self, sample_values: List[str]) -> bool:
        """Check if numeric values represent measurements/amounts"""
        if not sample_values:
            return False
            
        try:
            numeric_values = [float(v) for v in sample_values if v.replace('.', '').replace('-', '').isdigit()]
            if len(numeric_values) < 3:
                return False
                
            # Check for variance - measures usually have some variance
            variance = statistics.variance(numeric_values) if len(numeric_values) > 1 else 0
            mean_val = statistics.mean(numeric_values)
            
            # If coefficient of variation is reasonable, likely a measure
            if mean_val > 0:
                cv = (variance ** 0.5) / mean_val
                return 0.01 < cv < 10  # Some variance but not extreme
                
        except:
            pass
            
        return False
    
    def _calculate_pattern_scores(self, col_name: str, sample_values: List[str]) -> Dict[str, float]:
        """Calculate various pattern scores for the column"""
        scores = {}
        name_lower = col_name.lower()
        
        # Name-based pattern scores
        scores['identifier_score'] = 1.0 if name_lower.endswith('id') else 0.0
        scores['amount_score'] = 1.0 if any(term in name_lower for term in ['amount', 'total', 'sum', 'price', 'cost']) else 0.0
        scores['count_score'] = 1.0 if any(term in name_lower for term in ['count', 'number', 'qty']) else 0.0
        scores['name_score'] = 1.0 if 'name' in name_lower or 'title' in name_lower else 0.0
        
        # Value-based pattern scores
        if sample_values:
            # Email pattern
            email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
            scores['email_score'] = sum(1 for v in sample_values if email_pattern.match(v)) / len(sample_values)
            
            # Phone pattern (basic)
            phone_pattern = re.compile(r'^[\+]?[1-9][\d\-\(\)\s]{7,15}$')
            scores['phone_score'] = sum(1 for v in sample_values if phone_pattern.match(v)) / len(sample_values)
            
            # URL pattern
            url_pattern = re.compile(r'^https?://|www\.')
            scores['url_score'] = sum(1 for v in sample_values if url_pattern.match(v)) / len(sample_values)
        
        return scores
    
    def _classify_table_role(self, table_name: str, columns: List[ColumnProfile], sample_data: List[Dict]) -> TableRole:
        """Classify table role through structural analysis"""
        
        # Count column types
        type_counts = Counter(col.column_type for col in columns)
        fk_count = type_counts[ColumnType.FOREIGN_KEY]
        measure_count = type_counts[ColumnType.NUMERIC_MEASURE]
        temporal_count = type_counts[ColumnType.TEMPORAL]
        
        # Fact tables - many measures, temporal columns, foreign keys
        if measure_count >= 2 and temporal_count >= 1 and fk_count >= 1:
            return TableRole.FACT
        
        # Bridge tables - mostly foreign keys, few columns
        if fk_count >= 2 and len(columns) <= 5:
            return TableRole.BRIDGE
        
        # Lookup tables - mostly categorical, small size
        if (type_counts[ColumnType.CATEGORICAL] > len(columns) * 0.5 and 
            len(sample_data) < 100):
            return TableRole.LOOKUP
        
        # Log tables - timestamps, identifiers, text
        if (temporal_count >= 1 and 
            type_counts[ColumnType.TEXT_CONTENT] >= 1 and
            any('log' in table_name.lower() or 'audit' in table_name.lower() for _ in [1])):
            return TableRole.LOG
        
        # Configuration tables
        if (any('config' in table_name.lower() or 'setting' in table_name.lower() for _ in [1]) or
            len(sample_data) < 50):
            return TableRole.CONFIG
        
        # Default to dimension
        return TableRole.DIMENSION
    
    def _calculate_business_score(self, table_name: str, columns: List[ColumnProfile], sample_data: List[Dict]) -> float:
        """Calculate how important this table seems for business analysis"""
        score = 0.0
        
        # Size factor
        score += min(len(sample_data) / 1000, 5.0)  # More data = more important
        
        # Measure columns
        measure_cols = [col for col in columns if col.column_type == ColumnType.NUMERIC_MEASURE]
        score += len(measure_cols) * 2.0
        
        # Temporal columns
        temporal_cols = [col for col in columns if col.column_type == ColumnType.TEMPORAL] 
        score += len(temporal_cols) * 1.5
        
        # Foreign keys (indicates relationships)
        fk_cols = [col for col in columns if col.column_type == ColumnType.FOREIGN_KEY]
        score += len(fk_cols) * 1.0
        
        # Table role bonus
        if hasattr(self, '_table_role'):
            if self._table_role == TableRole.FACT:
                score += 10.0
            elif self._table_role == TableRole.DIMENSION:
                score += 5.0
        
        return score
    
    def _detect_relationships(self, table_profiles: List[TableProfile]) -> List[Dict[str, str]]:
        """Detect relationships between tables"""
        relationships = []
        
        # Build index of potential key columns
        key_columns = {}
        for table in table_profiles:
            for col in table.columns:
                if col.column_type == ColumnType.IDENTIFIER:
                    key_name = col.name.lower()
                    if key_name not in key_columns:
                        key_columns[key_name] = []
                    key_columns[key_name].append((table.name, col.name))
        
        # Find foreign key relationships
        for table in table_profiles:
            for col in table.columns:
                if col.column_type == ColumnType.FOREIGN_KEY:
                    fk_name = col.name.lower()
                    
                    # Look for matching primary keys
                    for key_name, tables_cols in key_columns.items():
                        if (fk_name.endswith(key_name) or key_name in fk_name) and len(tables_cols) > 0:
                            target_table, target_col = tables_cols[0]
                            if target_table != table.name:
                                relationships.append({
                                    "from_table": table.name,
                                    "from_column": col.name,
                                    "to_table": target_table,
                                    "to_column": target_col,
                                    "relationship_type": "foreign_key"
                                })
        
        return relationships
    
    def _identify_key_tables(self, table_profiles: List[TableProfile]) -> List[str]:
        """Identify the most important tables for analysis"""
        # Sort by business score
        sorted_tables = sorted(table_profiles, key=lambda t: t.business_score, reverse=True)
        
        # Take top tables, but ensure we have a mix of roles
        key_tables = []
        roles_seen = set()
        
        for table in sorted_tables:
            if len(key_tables) >= 8:  # Reasonable limit
                break
                
            # Always include high-scoring tables
            if table.business_score > 10 or len(key_tables) < 3:
                key_tables.append(table.name)
                roles_seen.add(table.table_role)
            # Include diverse table roles
            elif table.table_role not in roles_seen:
                key_tables.append(table.name)
                roles_seen.add(table.table_role)
        
        return key_tables
    
    def _generate_universal_analyses(self, table_profiles: List[TableProfile]) -> List[Dict[str, str]]:
        """Generate analysis suggestions that work for any domain"""
        suggestions = []
        
        # Find tables with measures for aggregation analysis
        fact_tables = [t for t in table_profiles if t.table_role == TableRole.FACT]
        for table in fact_tables[:3]:  # Top 3 fact tables
            measures = [col.name for col in table.columns if col.column_type == ColumnType.NUMERIC_MEASURE]
            temporal = [col.name for col in table.columns if col.column_type == ColumnType.TEMPORAL]
            
            if measures and temporal:
                suggestions.append({
                    "type": "trend_analysis",
                    "description": f"Analyze trends in {table.name} over time",
                    "table": table.name,
                    "measures": measures[:2],
                    "temporal": temporal[0]
                })
        
        # Distribution analysis for categorical columns
        for table in table_profiles:
            categorical_cols = [col.name for col in table.columns if col.column_type == ColumnType.CATEGORICAL]
            if categorical_cols:
                suggestions.append({
                    "type": "distribution_analysis", 
                    "description": f"Analyze distribution of categories in {table.name}",
                    "table": table.name,
                    "categories": categorical_cols[:2]
                })
        
        # Relationship analysis
        dimension_tables = [t for t in table_profiles if t.table_role == TableRole.DIMENSION]
        if fact_tables and dimension_tables:
            suggestions.append({
                "type": "relationship_analysis",
                "description": "Analyze relationships between entities",
                "fact_tables": [t.name for t in fact_tables[:2]], 
                "dimension_tables": [t.name for t in dimension_tables[:3]]
            })
        
        return suggestions
    
    def _infer_domain_characteristics(self, table_profiles: List[TableProfile]) -> List[str]:
        """Infer characteristics about the business domain"""
        insights = []
        
        # Analyze table names for domain clues
        table_names = [t.name.lower() for t in table_profiles]
        
        # Common domain patterns
        if any('user' in name or 'customer' in name or 'client' in name for name in table_names):
            insights.append("Customer/User management system detected")
        
        if any('order' in name or 'transaction' in name or 'payment' in name for name in table_names):
            insights.append("Transaction processing system detected")
        
        if any('product' in name or 'item' in name or 'inventory' in name for name in table_names):
            insights.append("Product/Inventory management system detected")
        
        if any('employee' in name or 'staff' in name for name in table_names):
            insights.append("Human resources system detected")
        
        # Analyze column patterns across tables
        all_columns = []
        for table in table_profiles:
            all_columns.extend([col.name.lower() for col in table.columns])
        
        column_counter = Counter(all_columns)
        
        if any('email' in col for col in all_columns):
            insights.append("Email communication capabilities present")
        
        if any('phone' in col for col in all_columns):
            insights.append("Phone/contact management present")
        
        if any('address' in col for col in all_columns):
            insights.append("Location/address management present")
        
        # Time-based analysis
        temporal_tables = len([t for t in table_profiles if any(col.column_type == ColumnType.TEMPORAL for col in t.columns)])
        if temporal_tables > len(table_profiles) * 0.7:
            insights.append("Time-series analysis capabilities - strong temporal tracking")
        
        return insights

    # Enhanced Business Intelligence Methods

    def _classify_business_domain(self, table_profiles: List[TableProfile]) -> Dict[str, Any]:
        """Advanced business domain classification with confidence scoring"""
        domain_indicators = {
            'financial_services': self._check_financial_patterns(table_profiles),
            'healthcare': self._check_healthcare_patterns(table_profiles),
            'retail_ecommerce': self._check_retail_patterns(table_profiles),
            'manufacturing': self._check_manufacturing_patterns(table_profiles),
            'human_resources': self._check_hr_patterns(table_profiles),
            'education': self._check_education_patterns(table_profiles),
            'logistics': self._check_logistics_patterns(table_profiles),
            'real_estate': self._check_real_estate_patterns(table_profiles),
            'telecommunications': self._check_telecom_patterns(table_profiles),
            'government': self._check_government_patterns(table_profiles)
        }

        # Find the highest scoring domain
        primary_domain = max(domain_indicators.items(), key=lambda x: x[1])

        # Calculate confidence (how much higher than average)
        scores = list(domain_indicators.values())
        avg_score = sum(scores) / len(scores)
        confidence = min(1.0, (primary_domain[1] - avg_score) / max(avg_score, 0.1))

        return {
            "primary_domain": primary_domain[0],
            "confidence": confidence,
            "all_scores": domain_indicators,
            "secondary_domains": [k for k, v in sorted(domain_indicators.items(), key=lambda x: x[1], reverse=True)[1:3]]
        }

    def _check_financial_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for financial services patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Financial table patterns
        financial_tables = ['account', 'transaction', 'payment', 'loan', 'credit', 'deposit', 'balance',
                          'portfolio', 'investment', 'fund', 'mortgage', 'insurance', 'claim', 'policy']
        score += sum(2.0 for pattern in financial_tables if any(pattern in name for name in table_names))

        # Financial column patterns
        financial_columns = ['account_number', 'balance', 'interest_rate', 'principal', 'payment_amount',
                           'transaction_amount', 'credit_limit', 'apr', 'maturity_date', 'risk_rating']
        score += sum(1.0 for pattern in financial_columns if any(pattern in col for col in all_columns))

        # Regulatory patterns
        regulatory_patterns = ['compliance', 'audit', 'regulation', 'kyc', 'aml', 'sox', 'basel']
        score += sum(1.5 for pattern in regulatory_patterns if any(pattern in name for name in table_names + all_columns))

        return min(10.0, score)

    def _check_healthcare_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for healthcare patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Healthcare table patterns
        healthcare_tables = ['patient', 'doctor', 'physician', 'nurse', 'appointment', 'diagnosis',
                           'treatment', 'medication', 'prescription', 'medical_record', 'hospital',
                           'clinic', 'department', 'billing', 'insurance_claim']
        score += sum(2.0 for pattern in healthcare_tables if any(pattern in name for name in table_names))

        # Healthcare column patterns
        healthcare_columns = ['patient_id', 'medical_record_number', 'diagnosis_code', 'procedure_code',
                            'medication_name', 'dosage', 'blood_pressure', 'temperature', 'symptoms']
        score += sum(1.0 for pattern in healthcare_columns if any(pattern in col for col in all_columns))

        # Medical coding patterns
        medical_codes = ['icd', 'cpt', 'hcpcs', 'snomed', 'loinc']
        score += sum(2.0 for pattern in medical_codes if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_retail_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for retail/e-commerce patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Retail table patterns
        retail_tables = ['product', 'inventory', 'order', 'customer', 'sale', 'purchase', 'supplier',
                        'category', 'brand', 'warehouse', 'shipment', 'cart', 'checkout', 'review']
        score += sum(2.0 for pattern in retail_tables if any(pattern in name for name in table_names))

        # Retail column patterns
        retail_columns = ['sku', 'product_id', 'price', 'cost', 'quantity_on_hand', 'reorder_level',
                         'order_total', 'discount', 'tax_amount', 'shipping_cost', 'rating']
        score += sum(1.0 for pattern in retail_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_manufacturing_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for manufacturing patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Manufacturing table patterns
        manufacturing_tables = ['production', 'assembly', 'part', 'component', 'material', 'machine',
                               'workstation', 'batch', 'lot', 'quality_control', 'defect', 'maintenance']
        score += sum(2.0 for pattern in manufacturing_tables if any(pattern in name for name in table_names))

        # Manufacturing column patterns
        manufacturing_columns = ['part_number', 'serial_number', 'batch_id', 'production_date',
                                'quality_score', 'defect_count', 'machine_id', 'cycle_time']
        score += sum(1.0 for pattern in manufacturing_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_hr_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for HR patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # HR table patterns
        hr_tables = ['employee', 'staff', 'payroll', 'salary', 'benefit', 'department', 'position',
                    'performance', 'review', 'training', 'attendance', 'leave', 'recruitment']
        score += sum(2.0 for pattern in hr_tables if any(pattern in name for name in table_names))

        # HR column patterns
        hr_columns = ['employee_id', 'hire_date', 'salary', 'job_title', 'department_id',
                     'manager_id', 'performance_rating', 'vacation_days', 'sick_leave']
        score += sum(1.0 for pattern in hr_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_education_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for education patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Education table patterns
        education_tables = ['student', 'teacher', 'instructor', 'course', 'class', 'grade', 'enrollment',
                           'transcript', 'degree', 'certification', 'exam', 'assignment', 'school']
        score += sum(2.0 for pattern in education_tables if any(pattern in name for name in table_names))

        # Education column patterns
        education_columns = ['student_id', 'course_id', 'grade', 'gpa', 'credit_hours',
                            'enrollment_date', 'graduation_date', 'major', 'transcript']
        score += sum(1.0 for pattern in education_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_logistics_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for logistics patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Logistics table patterns
        logistics_tables = ['shipment', 'delivery', 'route', 'vehicle', 'driver', 'warehouse',
                           'tracking', 'freight', 'carrier', 'package', 'manifest']
        score += sum(2.0 for pattern in logistics_tables if any(pattern in name for name in table_names))

        # Logistics column patterns
        logistics_columns = ['tracking_number', 'delivery_date', 'weight', 'dimensions',
                            'shipping_cost', 'origin', 'destination', 'carrier_id']
        score += sum(1.0 for pattern in logistics_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_real_estate_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for real estate patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Real estate table patterns
        real_estate_tables = ['property', 'listing', 'agent', 'buyer', 'seller', 'mortgage',
                             'appraisal', 'inspection', 'contract', 'closing']
        score += sum(2.0 for pattern in real_estate_tables if any(pattern in name for name in table_names))

        # Real estate column patterns
        real_estate_columns = ['property_id', 'listing_price', 'square_feet', 'bedrooms', 'bathrooms',
                              'lot_size', 'year_built', 'mls_number', 'commission']
        score += sum(1.0 for pattern in real_estate_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_telecom_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for telecommunications patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Telecom table patterns
        telecom_tables = ['subscriber', 'account', 'service', 'plan', 'usage', 'billing',
                         'network', 'cell_tower', 'call_detail', 'data_usage']
        score += sum(2.0 for pattern in telecom_tables if any(pattern in name for name in table_names))

        # Telecom column patterns
        telecom_columns = ['phone_number', 'subscriber_id', 'data_usage', 'minutes_used',
                          'roaming_charges', 'signal_strength', 'tower_id', 'bandwidth']
        score += sum(1.0 for pattern in telecom_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _check_government_patterns(self, table_profiles: List[TableProfile]) -> float:
        """Check for government patterns"""
        score = 0.0
        table_names = [t.name.lower() for t in table_profiles]
        all_columns = [col.name.lower() for t in table_profiles for col in t.columns]

        # Government table patterns
        government_tables = ['citizen', 'permit', 'license', 'tax', 'regulation', 'compliance',
                            'public_record', 'voting', 'census', 'benefit', 'social_security']
        score += sum(2.0 for pattern in government_tables if any(pattern in name for name in table_names))

        # Government column patterns
        government_columns = ['ssn', 'tax_id', 'permit_number', 'license_number', 'case_number',
                             'filing_date', 'expiration_date', 'status', 'jurisdiction']
        score += sum(1.0 for pattern in government_columns if any(pattern in col for col in all_columns))

        return min(10.0, score)

    def _generate_business_insights(self, table_profiles: List[TableProfile], domain: Dict[str, Any]) -> List[str]:
        """Generate business-specific insights based on domain classification"""
        insights = []
        primary_domain = domain["primary_domain"]
        confidence = domain["confidence"]

        insights.append(f"Domain Classification: {primary_domain.replace('_', ' ').title()} (confidence: {confidence:.1%})")

        if primary_domain == "financial_services":
            insights.extend(self._generate_financial_insights(table_profiles))
        elif primary_domain == "healthcare":
            insights.extend(self._generate_healthcare_insights(table_profiles))
        elif primary_domain == "retail_ecommerce":
            insights.extend(self._generate_retail_insights(table_profiles))
        elif primary_domain == "manufacturing":
            insights.extend(self._generate_manufacturing_insights(table_profiles))
        elif primary_domain == "human_resources":
            insights.extend(self._generate_hr_insights(table_profiles))
        else:
            insights.extend(self._generate_generic_business_insights(table_profiles))

        return insights

    def _generate_financial_insights(self, table_profiles: List[TableProfile]) -> List[str]:
        """Generate financial services specific insights"""
        insights = []

        # Look for core financial entities
        table_names = [t.name.lower() for t in table_profiles]

        if any('account' in name for name in table_names):
            insights.append("Account management system detected - likely core banking or investment platform")

        if any('transaction' in name for name in table_names):
            insights.append("Transaction processing capabilities - consider real-time fraud detection")

        if any('loan' in name for name in table_names):
            insights.append("Lending operations present - risk assessment and compliance monitoring recommended")

        if any('compliance' in name or 'audit' in name for name in table_names):
            insights.append("Regulatory compliance framework in place - ensure reporting capabilities")

        # Analyze temporal data for financial metrics
        temporal_tables = [t for t in table_profiles if any(col.column_type == ColumnType.TEMPORAL for col in t.columns)]
        if len(temporal_tables) > len(table_profiles) * 0.7:
            insights.append("Strong temporal tracking - excellent for trend analysis and regulatory reporting")

        return insights

    def _generate_healthcare_insights(self, table_profiles: List[TableProfile]) -> List[str]:
        """Generate healthcare specific insights"""
        insights = []

        table_names = [t.name.lower() for t in table_profiles]

        if any('patient' in name for name in table_names):
            insights.append("Patient management system - ensure HIPAA compliance and data privacy")

        if any('diagnosis' in name or 'procedure' in name for name in table_names):
            insights.append("Clinical data management - valuable for outcome analysis and quality metrics")

        if any('medication' in name or 'prescription' in name for name in table_names):
            insights.append("Pharmacy operations - drug interaction checking and dosage analysis possible")

        if any('billing' in name or 'insurance' in name for name in table_names):
            insights.append("Revenue cycle management - claims analysis and reimbursement optimization")

        return insights

    def _generate_retail_insights(self, table_profiles: List[TableProfile]) -> List[str]:
        """Generate retail/e-commerce specific insights"""
        insights = []

        table_names = [t.name.lower() for t in table_profiles]

        if any('inventory' in name for name in table_names):
            insights.append("Inventory management system - stockout analysis and demand forecasting possible")

        if any('customer' in name for name in table_names):
            insights.append("Customer data available - segmentation and lifetime value analysis recommended")

        if any('order' in name or 'sale' in name for name in table_names):
            insights.append("Sales transaction data - revenue analysis and seasonal trend detection")

        if any('review' in name or 'rating' in name for name in table_names):
            insights.append("Customer feedback system - sentiment analysis and product improvement insights")

        return insights

    def _generate_manufacturing_insights(self, table_profiles: List[TableProfile]) -> List[str]:
        """Generate manufacturing specific insights"""
        insights = []

        table_names = [t.name.lower() for t in table_profiles]

        if any('production' in name for name in table_names):
            insights.append("Production management system - efficiency metrics and capacity analysis possible")

        if any('quality' in name or 'defect' in name for name in table_names):
            insights.append("Quality control data - defect analysis and process improvement opportunities")

        if any('maintenance' in name for name in table_names):
            insights.append("Maintenance tracking - predictive maintenance and downtime analysis potential")

        if any('material' in name or 'component' in name for name in table_names):
            insights.append("Supply chain data - vendor analysis and cost optimization opportunities")

        return insights

    def _generate_hr_insights(self, table_profiles: List[TableProfile]) -> List[str]:
        """Generate HR specific insights"""
        insights = []

        table_names = [t.name.lower() for t in table_profiles]

        if any('employee' in name for name in table_names):
            insights.append("Employee management system - workforce analytics and retention analysis possible")

        if any('performance' in name for name in table_names):
            insights.append("Performance management data - talent development and promotion analytics")

        if any('payroll' in name or 'salary' in name for name in table_names):
            insights.append("Compensation data - pay equity analysis and budget planning capabilities")

        if any('training' in name for name in table_names):
            insights.append("Learning and development tracking - skill gap analysis and training ROI")

        return insights

    def _generate_generic_business_insights(self, table_profiles: List[TableProfile]) -> List[str]:
        """Generate generic business insights when domain is unclear"""
        insights = []

        # Analyze table relationships
        fact_tables = [t for t in table_profiles if t.table_role == TableRole.FACT]
        dimension_tables = [t for t in table_profiles if t.table_role == TableRole.DIMENSION]

        if fact_tables and dimension_tables:
            insights.append("Star/snowflake schema pattern detected - well-structured for analytical reporting")

        # Analyze data complexity
        avg_columns = sum(len(t.columns) for t in table_profiles) / max(len(table_profiles), 1)
        if avg_columns > 15:
            insights.append("Complex data structure - consider data governance and documentation standards")

        # Analyze temporal capabilities
        temporal_tables = [t for t in table_profiles if any(col.column_type == ColumnType.TEMPORAL for col in t.columns)]
        if len(temporal_tables) > len(table_profiles) * 0.5:
            insights.append("Strong temporal data - excellent foundation for trend analysis and forecasting")

        return insights

    def _generate_advanced_kpi_recommendations(self, table_profiles: List[TableProfile], domain: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate advanced KPI recommendations based on domain and data structure"""
        recommendations = []
        primary_domain = domain["primary_domain"]

        # Domain-specific KPI recommendations
        if primary_domain == "financial_services":
            recommendations.extend([
                {"kpi": "Customer Acquisition Cost", "calculation": "Total acquisition costs / New customers", "business_value": "Optimize marketing spend"},
                {"kpi": "Net Interest Margin", "calculation": "(Interest Income - Interest Expense) / Average Earning Assets", "business_value": "Profitability measurement"},
                {"kpi": "Credit Loss Ratio", "calculation": "Credit Losses / Total Loans", "business_value": "Risk management"},
                {"kpi": "Cost-to-Income Ratio", "calculation": "Operating Expenses / Operating Income", "business_value": "Operational efficiency"}
            ])
        elif primary_domain == "retail_ecommerce":
            recommendations.extend([
                {"kpi": "Customer Lifetime Value", "calculation": "Average Order Value × Purchase Frequency × Customer Lifespan", "business_value": "Customer investment strategy"},
                {"kpi": "Inventory Turnover", "calculation": "Cost of Goods Sold / Average Inventory", "business_value": "Inventory optimization"},
                {"kpi": "Cart Abandonment Rate", "calculation": "Abandoned Carts / Total Carts", "business_value": "Conversion optimization"},
                {"kpi": "Return on Ad Spend", "calculation": "Revenue from Ads / Ad Spend", "business_value": "Marketing effectiveness"}
            ])
        elif primary_domain == "manufacturing":
            recommendations.extend([
                {"kpi": "Overall Equipment Effectiveness", "calculation": "Availability × Performance × Quality", "business_value": "Production optimization"},
                {"kpi": "First Pass Yield", "calculation": "Units passing first inspection / Total units produced", "business_value": "Quality control"},
                {"kpi": "Production Efficiency", "calculation": "Actual Output / Planned Output", "business_value": "Capacity utilization"},
                {"kpi": "Mean Time Between Failures", "calculation": "Total operating time / Number of failures", "business_value": "Maintenance planning"}
            ])

        # Generic KPIs based on data structure
        numeric_tables = [t for t in table_profiles if any(col.column_type == ColumnType.NUMERIC_MEASURE for col in t.columns)]
        if numeric_tables:
            recommendations.append({
                "kpi": "Growth Rate Analysis",
                "calculation": "(Current Period - Previous Period) / Previous Period × 100",
                "business_value": "Trend identification"
            })

        temporal_tables = [t for t in table_profiles if any(col.column_type == ColumnType.TEMPORAL for col in t.columns)]
        if len(temporal_tables) > len(table_profiles) * 0.5:
            recommendations.append({
                "kpi": "Seasonal Variance",
                "calculation": "Percentage deviation from baseline by time period",
                "business_value": "Demand forecasting"
            })

        return recommendations


class UniversalQueryOrchestrator(LoggerMixin):
    """
    Creates and executes dynamic query plans for any schema
    """
    
    def __init__(self, client: GlobalMCPClient):
        self.client = client
        self.analyzer = UniversalSchemaAnalyzer(client)
    
    async def create_analysis_plan(self, schema_name: str, request: str) -> List[QueryPlan]:
        """Create a dynamic analysis plan based on schema analysis and request"""
        
        # Analyze the schema
        schema_analysis = await self.analyzer.analyze_schema(schema_name)
        
        # Classify the request
        request_type = self._classify_request(request)
        
        # Generate appropriate plans
        plans = []
        
        if request_type == "overview" or request_type == "dashboard":
            plans.extend(await self._create_overview_plan(schema_analysis))
        elif request_type == "detailed":
            plans.extend(await self._create_detailed_plan(schema_analysis))
        elif request_type == "relationships":
            plans.extend(await self._create_relationship_plan(schema_analysis))
        else:
            # Default comprehensive analysis
            plans.extend(await self._create_comprehensive_plan(schema_analysis))
        
        return plans
    
    def _classify_request(self, request: str) -> str:
        """Classify the request type to determine analysis approach"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["overview", "summary", "dashboard", "metrics"]):
            return "overview"
        elif any(word in request_lower for word in ["detailed", "deep", "comprehensive", "complete"]):
            return "detailed" 
        elif any(word in request_lower for word in ["relationship", "connection", "link", "association"]):
            return "relationships"
        elif any(word in request_lower for word in ["trend", "time", "temporal", "historical"]):
            return "temporal"
        else:
            return "comprehensive"
    
    async def _create_overview_plan(self, schema_analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Create plans for high-level overview"""
        plans = []
        key_tables = schema_analysis["key_tables"][:5]  # Top 5 tables
        
        # Schema overview
        plans.append(QueryPlan(
            id="schema_overview",
            description="Schema structure overview",
            priority=1,
            tool_name="get_all_tables",
            parameters={"schemaName": schema_analysis["schema_name"]},
            expected_insight="Understanding of overall schema structure"
        ))
        
        # Key table summaries
        for i, table_name in enumerate(key_tables):
            plans.append(QueryPlan(
                id=f"table_summary_{i}",
                description=f"Data summary for {table_name}",
                priority=2,
                tool_name="query_table_records", 
                parameters={
                    "tableName": table_name,
                    "columns": "*",
                    "limit": 5
                },
                expected_insight=f"Sample data and structure of {table_name}"
            ))
        
        return plans
    
    async def _create_detailed_plan(self, schema_analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Create plans for detailed analysis"""
        plans = []
        
        # Include overview first
        plans.extend(await self._create_overview_plan(schema_analysis))
        
        # Detailed analysis of each key table
        for table_profile in schema_analysis.get("table_profiles", []):
            if table_profile["name"] in schema_analysis["key_tables"]:
                
                # Get larger sample
                plans.append(QueryPlan(
                    id=f"detailed_{table_profile['name']}",
                    description=f"Detailed analysis of {table_profile['name']}",
                    priority=3,
                    tool_name="query_table_records",
                    parameters={
                        "tableName": table_profile["name"],
                        "columns": "*",
                        "limit": 20
                    },
                    expected_insight=f"Comprehensive view of {table_profile['name']} data patterns"
                ))
        
        return plans
    
    async def _create_relationship_plan(self, schema_analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Create plans focusing on relationships"""
        plans = []
        
        relationships = schema_analysis.get("relationships", [])
        
        # Analyze each relationship
        for i, rel in enumerate(relationships[:5]):  # Top 5 relationships
            plans.append(QueryPlan(
                id=f"relationship_{i}",
                description=f"Analyze relationship: {rel['from_table']} -> {rel['to_table']}",
                priority=2,
                tool_name="query_table_records",
                parameters={
                    "tableName": rel["from_table"],
                    "columns": f"{rel['from_column']},*",
                    "limit": 10
                },
                expected_insight=f"Understanding relationship between {rel['from_table']} and {rel['to_table']}"
            ))
        
        return plans
    
    async def _create_comprehensive_plan(self, schema_analysis: Dict[str, Any]) -> List[QueryPlan]:
        """Create comprehensive analysis plan"""
        plans = []
        
        # Start with overview
        plans.extend(await self._create_overview_plan(schema_analysis))
        
        # Add relationship analysis
        if schema_analysis.get("relationships"):
            plans.extend(await self._create_relationship_plan(schema_analysis))
        
        # Add suggested analyses
        for i, suggestion in enumerate(schema_analysis.get("suggested_analyses", [])[:3]):
            if suggestion["type"] == "trend_analysis":
                plans.append(QueryPlan(
                    id=f"trend_{i}",
                    description=suggestion["description"],
                    priority=4,
                    tool_name="query_table_records",
                    parameters={
                        "tableName": suggestion["table"],
                        "columns": ",".join(suggestion["measures"] + [suggestion["temporal"]]),
                        "limit": 50,
                        "orderBy": suggestion["temporal"]
                    },
                    expected_insight="Time-based trends and patterns"
                ))
        
        return plans
    
    async def execute_plan(self, plans: List[QueryPlan]) -> List[AnalysisResult]:
        """Execute analysis plans and return results"""
        results = []
        
        # Sort by priority
        sorted_plans = sorted(plans, key=lambda p: p.priority)
        
        for plan in sorted_plans:
            self.logger.info(f"Executing plan: {plan.description}")
            start_time = time.time()
            
            try:
                # Execute the query
                data = await self.client.call_tool(plan.tool_name, plan.parameters)
                execution_time = time.time() - start_time
                
                # Format output
                formatted_output = await self._format_result(plan, data)
                
                # Extract insights
                insights = self._extract_insights(plan, data)
                
                results.append(AnalysisResult(
                    plan_id=plan.id,
                    success=True,
                    execution_time=execution_time,
                    data=data,
                    formatted_output=formatted_output,
                    insights=insights
                ))
                
            except Exception as e:
                execution_time = time.time() - start_time
                results.append(AnalysisResult(
                    plan_id=plan.id,
                    success=False,
                    execution_time=execution_time,
                    error=str(e)
                ))
                self.logger.error(f"Plan execution failed: {e}")
        
        return results
    
    async def _format_result(self, plan: QueryPlan, data: Any) -> str:
        """Format result for human consumption"""
        if not data:
            return "No data returned"
        
        output = f"## {plan.description}\n\n"
        
        try:
            if isinstance(data, list) and data:
                if hasattr(data[0], 'text'):
                    result_data = json.loads(data[0].text)
                    
                    if "tables" in result_data:
                        # Schema overview format
                        tables = result_data["tables"]
                        output += f"**Schema contains {len(tables)} tables**\n\n"
                        output += "| Table Name | Rows | Type |\n"
                        output += "|------------|------|------|\n"
                        
                        for table in tables[:10]:
                            name = table.get("TABLE_NAME", "Unknown")
                            rows = table.get("NUM_ROWS", "N/A")
                            output += f"| {name} | {rows} | Table |\n"
                            
                    elif "records" in result_data:
                        # Data records format
                        records = result_data["records"]
                        if records:
                            output += f"**Found {len(records)} records**\n\n"
                            
                            # Create table
                            headers = list(records[0].keys())
                            output += "| " + " | ".join(headers) + " |\n"
                            output += "|" + "|".join(["---"] * len(headers)) + "|\n"
                            
                            for record in records[:5]:
                                values = [str(record.get(h, ""))[:30] for h in headers]
                                output += "| " + " | ".join(values) + " |\n"
                                
                            if len(records) > 5:
                                output += f"\n*Showing 5 of {len(records)} records*\n"
                    
                    else:
                        output += f"Raw data: {str(result_data)[:500]}...\n"
                        
        except Exception as e:
            output += f"Error formatting result: {e}\n"
            
        return output
    
    def _extract_insights(self, plan: QueryPlan, data: Any) -> List[str]:
        """Extract business insights from the data"""
        insights = []
        
        try:
            if isinstance(data, list) and data:
                result_data = json.loads(data[0].text)
                
                if "records" in result_data:
                    records = result_data["records"]
                    if records:
                        insights.append(f"Dataset contains {len(records)} records")
                        
                        # Analyze data quality
                        if records:
                            sample = records[0]
                            null_fields = sum(1 for v in sample.values() if v is None or v == "")
                            if null_fields > 0:
                                insights.append(f"Data quality note: {null_fields} empty fields detected in sample")
                            
                            # Identify potential key fields
                            for key, value in sample.items():
                                if isinstance(value, (int, float)) and value > 1000:
                                    insights.append(f"High numeric values in {key} - potential amount/quantity field")
                
        except Exception as e:
            insights.append(f"Analysis error: {e}")
            
        return insights


class UniversalAutoProcessor(LoggerMixin):
    """
    Main processor that coordinates universal schema analysis
    """
    
    def __init__(self, client: GlobalMCPClient):
        self.client = client
        self.orchestrator = UniversalQueryOrchestrator(client)
    
    async def process_request(self, request: str, schema_name: str) -> str:
        """Process any request against any schema"""
        self.logger.info(f"Processing universal request: {request} for schema: {schema_name}")
        
        try:
            # Create analysis plan
            plans = await self.orchestrator.create_analysis_plan(schema_name, request)
            
            if not plans:
                return "Unable to generate analysis plans for this request."
            
            # Execute plans
            results = await self.orchestrator.execute_plan(plans)
            
            # Generate final report
            report = self._generate_report(request, schema_name, results)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return f"Analysis failed: {str(e)}"
    
    def _generate_report(self, request: str, schema_name: str, results: List[AnalysisResult]) -> str:
        """Generate comprehensive analysis report"""
        
        report = f"# Universal Schema Analysis\n\n"
        report += f"**Request:** {request}\n"
        report += f"**Schema:** {schema_name}\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Executive Summary
        successful = len([r for r in results if r.success])
        total = len(results)
        total_time = sum(r.execution_time for r in results)
        
        report += f"## Executive Summary\n\n"
        report += f"- Executed {successful}/{total} analyses successfully\n"
        report += f"- Total execution time: {total_time:.2f} seconds\n"
        report += f"- Average query time: {total_time/max(total,1):.2f} seconds\n\n"
        
        # Detailed Results
        report += f"## Analysis Results\n\n"
        
        for result in results:
            if result.success:
                report += result.formatted_output + "\n"
                
                if result.insights:
                    report += f"**Key Insights:**\n"
                    for insight in result.insights:
                        report += f"- {insight}\n"
                    report += "\n"
            else:
                report += f"### Analysis Failed: {result.plan_id}\n"
                report += f"Error: {result.error}\n\n"
        
        # Recommendations
        report += f"## Recommendations\n\n"
        report += self._generate_recommendations(results)
        
        return report
    
    def _generate_recommendations(self, results: List[AnalysisResult]) -> str:
        """Generate universal recommendations"""
        recommendations = ""
        
        successful_results = [r for r in results if r.success]
        
        if successful_results:
            recommendations += "- **Data Quality**: Review any empty fields or missing values identified\n"
            recommendations += "- **Performance**: Consider indexing on frequently queried columns\n"
            recommendations += "- **Analysis**: Use the identified key tables for regular reporting\n"
            recommendations += "- **Monitoring**: Set up alerts for important metrics in key tables\n"
        
        if len(successful_results) < len(results):
            recommendations += "- **Troubleshooting**: Investigate failed queries for access or data issues\n"
        
        recommendations += "\n*This analysis is completely dynamic and works with any schema structure.*\n"
        
        return recommendations
