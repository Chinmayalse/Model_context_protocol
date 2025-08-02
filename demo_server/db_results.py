"""
Results management module for PostgreSQL database interaction.

This module provides a ResultManager class for managing processing results,
with support for batch operations, status tracking, and file management.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum, auto
from pathlib import Path

from db import get_db_cursor, get_db_connection
from psycopg2.extras import Json, DictCursor, execute_batch

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ProcessingStatus(str, Enum):
    """Status of a processing result."""
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    PARTIAL_SUCCESS = 'partial_success'
    CANCELLED = 'cancelled'

@dataclass
class ProcessingResult:
    """Data class representing a processing result."""
    id: int
    user_id: int
    filename: str
    original_filename: str
    processing_type: str
    tool_used: str
    status: str
    result_data: Dict[str, Any]
    error_message: Optional[str] = None
    chat_id: Optional[str] = None
    batch_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    is_deleted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        for field in ['created_at', 'updated_at']:
            if result[field]:
                result[field] = result[field].isoformat()
        return result
    
    @classmethod
    def from_db_row(cls, row: Dict) -> 'ProcessingResult':
        """Create a ProcessingResult from a database row."""
        try:
            # Debug: Log the row keys and values
            logger.debug(f"Creating ProcessingResult from row with keys: {list(row.keys())}")
            
            # Ensure result_data is a dictionary
            result_data = row.get('result_data', {})
            if isinstance(result_data, str):
                try:
                    result_data = json.loads(result_data)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse result_data as JSON: {result_data}")
                    result_data = {}
            
            # Handle potential None values
            created_at = row.get('created_at')
            updated_at = row.get('updated_at')
            
            # Create and return the ProcessingResult
            return cls(
                id=row.get('id'),
                user_id=row.get('user_id'),
                filename=row.get('filename', ''),
                original_filename=row.get('original_filename', ''),
                processing_type=row.get('processing_type', ''),
                tool_used=row.get('tool_used', ''),
                status=row.get('status', ProcessingStatus.COMPLETED),
                result_data=result_data,
                error_message=row.get('error_message'),
                chat_id=row.get('chat_id'),
                batch_id=row.get('batch_id'),
                created_at=created_at,
                updated_at=updated_at,
                is_deleted=row.get('is_deleted', False)
            )
        except Exception as e:
            logger.error(f"Error creating ProcessingResult from row: {e}", exc_info=True)
            logger.error(f"Row data: {row}")
            raise

class ResultManager:
    @staticmethod
    def save_result(
        user_id: int,
        filename: str,
        original_filename: str,
        processing_type: str,
        tool_used: str,
        result_data: Dict[str, Any],
        status: str = ProcessingStatus.COMPLETED,
        error_message: Optional[str] = None,
        chat_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        connection=None
    ) -> int:
        """
        Save a processing result to the database.
        
        Args:
            user_id: The ID of the user who owns this result
            filename: The filename for the result JSON
            original_filename: The original medical report filename
            processing_type: Type of processing ('summary', 'process', etc.)
            tool_used: Tool used for processing
            result_data: The result data as a dictionary
            status: Status of the processing (default: 'completed')
            error_message: Optional error message if processing failed
            chat_id: Optional chat ID where this result was generated
            batch_id: Optional batch ID for batch processing
            connection: Optional database connection to use (for transactions)
            
        Returns:
            The ID of the inserted result
            
        Raises:
            ValueError: If required parameters are missing
            Exception: If database operation fails
        """
        if not all([user_id, filename, original_filename, processing_type, tool_used]):
            raise ValueError("Missing required parameters")
            
        now = datetime.now(timezone.utc)
        
        # Prepare the result data
        if not isinstance(result_data, dict):
            result_data = {'data': result_data}
            
        # Add status to result data if not present
        if 'status' not in result_data:
            result_data['status'] = status
            
        if error_message and 'error' not in result_data:
            result_data['error'] = error_message
        
        query = """
            INSERT INTO results 
            (user_id, filename, original_filename, processing_type, tool_used, 
             status, result_data, chat_id, batch_id, 
             created_at, is_deleted)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """
        
        # If there's an error message, include it in the result_data
        if error_message:
            result_data['error'] = error_message
            
        params = (
            user_id, filename, original_filename, processing_type, tool_used,
            status, Json(result_data), chat_id, batch_id,
            now, False
        )
        
        try:
            if connection:
                # Use the provided connection for transactions
                with connection.cursor() as cur:
                    cur.execute(query, params)
                    result = cur.fetchone()
                    result_id = result[0]  # Access first column by index
                    if not connection.get_transaction_status():
                        connection.commit()
                    return result_id
            else:
                # Create a new connection with DictCursor to get results as dictionaries
                with get_db_cursor(commit=True, cursor_factory=DictCursor) as cur:
                    cur.execute(query, params)
                    result = cur.fetchone()
                    # Handle both tuple and dictionary access
                    if isinstance(result, dict):
                        return result['id']
                    else:
                        return result[0]  # First column is the ID
                    
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def get_result_by_id(result_id: int, user_id: Optional[int] = None) -> Optional[ProcessingResult]:
        """
        Get a result by its ID.
        
        Args:
            result_id: The ID of the result to retrieve
            user_id: Optional user ID for security filtering
            
        Returns:
            ProcessingResult object if found, None otherwise
        """
        query = """
            SELECT * FROM results 
            WHERE id = %s AND is_deleted = FALSE
        """
        params = [result_id]
        
        if user_id is not None:
            query += " AND user_id = %s"
            params.append(user_id)
            
        try:
            # Use DictCursor to get results as dictionaries
            with get_db_cursor(cursor_factory=DictCursor) as cur:
                logger.debug(f"Executing query: {query} with params: {params}")
                cur.execute(query, tuple(params))
                row = cur.fetchone()
                
                if not row:
                    logger.debug(f"No result found with ID {result_id}")
                    return None
                    
                # Convert row to dict if it's not already
                row_dict = dict(row) if not isinstance(row, dict) else row
                return ProcessingResult.from_db_row(row_dict)
                
        except Exception as e:
            logger.error(f"Error getting result {result_id}: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def get_result_by_filename(filename: str, user_id: Optional[int] = None) -> Optional[ProcessingResult]:
        """
        Get a result by its filename.
        
        Args:
            filename: The filename of the result
            user_id: Optional user ID for security filtering
            
        Returns:
            ProcessingResult object if found, None otherwise
        """
        query = """
            SELECT * FROM results 
            WHERE filename = %s AND is_deleted = FALSE
        """
        params = [filename]
        
        if user_id is not None:
            query += " AND user_id = %s"
            params.append(user_id)
            
        try:
            # Use DictCursor to get results as dictionaries
            with get_db_cursor(cursor_factory=DictCursor) as cur:
                logger.debug(f"Executing query: {query} with params: {params}")
                cur.execute(query, tuple(params))
                row = cur.fetchone()
                
                if not row:
                    logger.debug(f"No result found with filename {filename}")
                    return None
                    
                # Convert row to dict if it's not already
                row_dict = dict(row) if not isinstance(row, dict) else row
                return ProcessingResult.from_db_row(row_dict)
                
        except Exception as e:
            logger.error(f"Error getting result for filename {filename}: {str(e)}", exc_info=True)
            return None
    
    @staticmethod
    def get_results_by_user(
        user_id: int, 
        limit: int = 100, 
        offset: int = 0,
        batch_id: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[ProcessingResult]:
        """
        Get results for a specific user with pagination and filtering.
        
        Args:
            user_id: The ID of the user
            limit: Maximum number of results to return
            offset: Number of results to skip
            batch_id: Optional filter by batch ID
            status: Optional filter by status
            
        Returns:
            List of ProcessingResult objects
        """
        query = """
            SELECT * FROM results 
            WHERE user_id = %s AND is_deleted = FALSE
        """
        params = [user_id]
        
        if batch_id:
            query += " AND batch_id = %s"
            params.append(batch_id)
            
        if status:
            query += " AND status = %s"
            params.append(status)
            
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        try:
            # Use DictCursor to get results as dictionaries
            with get_db_cursor(cursor_factory=DictCursor) as cur:
                logger.debug(f"Executing query: {query} with params: {params}")
                cur.execute(query, tuple(params))
                
                rows = cur.fetchall()
                if not rows:
                    logger.debug(f"No results found for user {user_id}")
                    return []
                
                # Process each row
                results = []
                for row in rows:
                    try:
                        # Convert row to dict if it's not already
                        row_dict = dict(row) if not isinstance(row, dict) else row
                        results.append(ProcessingResult.from_db_row(row_dict))
                    except Exception as e:
                        logger.error(f"Error processing row: {e}", exc_info=True)
                        logger.error(f"Row data: {row}")
                        continue
                        
                logger.debug(f"Returning {len(results)} results for user {user_id}")
                return results
                
        except Exception as e:
            logger.error(f"Error getting results for user {user_id}: {str(e)}", exc_info=True)
            return []
    
    @staticmethod
    def get_results_by_batch(
        batch_id: str, 
        user_id: Optional[int] = None,
        limit: int = 1000,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[ProcessingResult]:
        """
        Get all results for a specific batch with pagination and filtering.
        
        Args:
            batch_id: The ID of the batch
            user_id: Optional user ID for security filtering
            limit: Maximum number of results to return
            offset: Number of results to skip
            status: Optional status filter
            
        Returns:
            List of ProcessingResult objects
        """
        if not batch_id:
            return []
            
        query = """
            SELECT * FROM results 
            WHERE batch_id = %s AND is_deleted = FALSE
        """
        params = [batch_id]
        
        if user_id is not None:
            query += " AND user_id = %s"
            params.append(user_id)
            
        if status:
            query += " AND status = %s"
            params.append(status)
            
        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        try:
            # Use DictCursor to get results as dictionaries
            with get_db_cursor(cursor_factory=DictCursor) as cur:
                logger.debug(f"Executing query: {query} with params: {params}")
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                
                if not rows:
                    logger.debug(f"No results found for batch {batch_id}")
                    return []
                
                # Process each row
                results = []
                for row in rows:
                    try:
                        # Convert row to dict if it's not already
                        row_dict = dict(row) if not isinstance(row, dict) else row
                        results.append(ProcessingResult.from_db_row(row_dict))
                    except Exception as e:
                        logger.error(f"Error processing row: {e}", exc_info=True)
                        logger.error(f"Row data: {row}")
                        continue
                        
                logger.debug(f"Returning {len(results)} results for batch {batch_id}")
                return results
                
        except Exception as e:
            logger.error(f"Error getting results for batch {batch_id}: {str(e)}", exc_info=True)
            return []
    
    @staticmethod
    def get_batch_summary(batch_id: str, user_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a detailed summary of a batch processing job.
        
        Args:
            batch_id: The ID of the batch
            user_id: Optional user ID for security filtering
            
        Returns:
            Dictionary with batch summary information
        """
        # First get basic batch info from the batch table if it exists
        batch_info = {}
        try:
            with get_db_cursor(cursor_factory=DictCursor) as cur:
                logger.debug(f"Fetching batch info for batch {batch_id}")
                cur.execute(
                    """
                    SELECT * FROM batches 
                    WHERE id = %s AND (user_id = %s OR %s IS NULL)
                    """,
                    (batch_id, user_id, user_id)
                )
                row = cur.fetchone()
                if row:
                    batch_info = dict(row) if not isinstance(row, dict) else row
        except Exception as e:
            logger.warning(f"Could not fetch batch info: {str(e)}", exc_info=True)
        
        # Then get statistics about the results
        stats_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) as processing,
                MIN(created_at) as started_at,
                MAX(updated_at) as last_updated
            FROM results 
            WHERE batch_id = %s
        """
        
        params = [batch_id]
        if user_id is not None:
            stats_query += " AND user_id = %s"
            params.append(user_id)
        
        stats = {
            'total': 0,
            'completed': 0,
            'failed': 0,
            'processing': 0,
            'started_at': None,
            'last_updated': None
        }
        
        try:
            with get_db_cursor(cursor_factory=DictCursor) as cur:
                logger.debug(f"Fetching stats for batch {batch_id}")
                cur.execute(stats_query, tuple(params))
                row = cur.fetchone()
                if row:
                    row_dict = dict(row) if not isinstance(row, dict) else row
                    stats.update({
                        'total': row_dict.get('total', 0) or 0,
                        'completed': row_dict.get('completed', 0) or 0,
                        'failed': row_dict.get('failed', 0) or 0,
                        'processing': row_dict.get('processing', 0) or 0,
                        'started_at': row_dict.get('started_at'),
                        'last_updated': row_dict.get('last_updated')
                    })
        except Exception as e:
            logger.error(f"Error getting batch stats: {str(e)}", exc_info=True)
        
        # Calculate duration if we have both start and end times
        duration_seconds = 0
        if stats['started_at'] and stats['last_updated']:
            try:
                duration_seconds = (stats['last_updated'] - stats['started_at']).total_seconds()
            except Exception as e:
                logger.warning(f"Could not calculate duration: {str(e)}")
        
        # Format duration as HH:MM:SS
        hours, remainder = divmod(int(duration_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Determine overall status
        status = 'completed'  # Default status
        if stats['failed'] > 0 and stats['completed'] > 0:
            status = 'partial_success'
        elif stats['failed'] > 0:
            status = 'failed'
        elif stats['processing'] > 0:
            status = 'processing'
        elif stats['completed'] == 0:
            status = 'pending'
        
        return {
            'batch_id': batch_id,
            'batch_info': batch_info,
            'status': status,
            'stats': {
                'total': stats['total'],
                'completed': stats['completed'],
                'failed': stats['failed'],
                'processing': stats['processing'],
                'started_at': stats['started_at'].isoformat() if stats['started_at'] else None,
                'last_updated': stats['last_updated'].isoformat() if stats['last_updated'] else None,
                'duration_seconds': duration_seconds,
                'duration': duration_str,
                'total_files': stats['total'],
                'progress': int((stats['completed'] + stats['failed']) / stats['total'] * 100) if stats['total'] > 0 else 0,
                'processed': stats['completed'] + stats['failed']
            }
        }
            
    @staticmethod
    def delete_result(result_id: int, user_id: int) -> bool:
        """
        Mark a result as deleted.
        
        Args:
            result_id: The ID of the result to delete
            user_id: The ID of the user who owns the result
            
        Returns:
            True if the result was deleted, False otherwise
        """
        if not result_id or not user_id:
            return False
            
        query = """
            UPDATE results 
            SET is_deleted = TRUE
            WHERE id = %s AND user_id = %s
            RETURNING id
        """
        
        try:
            with get_db_cursor(commit=True) as cur:
                cur.execute(query, (result_id, user_id))
                return cur.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting result {result_id}: {str(e)}", exc_info=True)
            return False
    
    @staticmethod
    def get_recent_results(
        limit: int = 10, 
        user_id: Optional[int] = None,
        processing_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the most recent results with optional filtering.
        
        Args:
            limit: Maximum number of results to return (max 100)
            user_id: Optional user ID to filter by
            processing_type: Optional processing type to filter by
            
        Returns:
            List of result dictionaries
        """
        # Validate and sanitize inputs
        try:
            limit = max(1, min(100, int(limit)))
        except (ValueError, TypeError):
            limit = 10
            
        query = """
            SELECT * FROM results 
            WHERE is_deleted = FALSE
        """
        params = []
        
        if user_id is not None:
            query += " AND user_id = %s"
            params.append(user_id)
            
        if processing_type:
            query += " AND processing_type = %s"
            params.append(processing_type)
            
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        
        try:
            with get_db_cursor(cursor_factory=DictCursor) as cur:
                cur.execute(query, tuple(params))
                results = []
                
                for row in cur.fetchall():
                    result = {
                        'id': row['id'],
                        'filename': row['filename'],
                        'original_filename': row['original_filename'],
                        'processing_type': row['processing_type'],
                        'tool_used': row['tool_used'],
                        'status': row.get('status', ProcessingStatus.COMPLETED),
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'batch_id': row.get('batch_id')
                    }
                    
                    # Include additional metadata if available
                    if row.get('result_data'):
                        result_data = dict(row['result_data'])
                        result.update({
                            'success': result_data.get('success', True),
                            'error': result_data.get('error'),
                            'extract_text': result_data.get('extract_text'),
                            'summary': result_data.get('summary')
                        })
                    
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error getting recent results: {str(e)}", exc_info=True)
            return []
    
    @staticmethod
    def get_processing_stats(
        user_id: Optional[int] = None,
        days: int = 30,
        group_by: str = 'day'
    ) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Args:
            user_id: Optional user ID to filter by
            days: Number of days to look back
            group_by: Grouping interval ('day', 'week', 'month')
            
        Returns:
            Dictionary with processing statistics
        """
        if group_by not in ['day', 'week', 'month']:
            group_by = 'day'
            
        interval = {
            'day': '1 day',
            'week': '1 week',
            'month': '1 month'
        }[group_by]
        
        query = f"""
            SELECT 
                date_trunc(%s, created_at) as period,
                COUNT(*) as total,
                COUNT(CASE WHEN status = %s THEN 1 END) as completed,
                COUNT(CASE WHEN status = %s THEN 1 END) as failed,
                COUNT(CASE WHEN status = %s THEN 1 END) as processing
            FROM results
            WHERE 
                created_at >= NOW() - INTERVAL '%s days' 
                AND is_deleted = FALSE
        """
        
        params = [group_by, ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.PROCESSING, days]
        
        if user_id is not None:
            query += " AND user_id = %s"
            params.append(user_id)
            
        query += """
            GROUP BY period
            ORDER BY period
        """
        
        try:
            with get_db_cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
                
                # Format the results
                stats = {
                    'periods': [],
                    'totals': [],
                    'completed': [],
                    'failed': [],
                    'processing': []
                }
                
                for row in rows:
                    period = row['period'].strftime('%Y-%m-%d')
                    stats['periods'].append(period)
                    stats['totals'].append(row['total'])
                    stats['completed'].append(row['completed'])
                    stats['failed'].append(row['failed'])
                    stats['processing'].append(row['processing'])
                
                # Add summary
                stats['summary'] = {
                    'total': sum(stats['totals']) if stats['totals'] else 0,
                    'completed': sum(stats['completed']) if stats['completed'] else 0,
                    'failed': sum(stats['failed']) if stats['failed'] else 0,
                    'processing': sum(stats['processing']) if stats['processing'] else 0,
                    'success_rate': (
                        round((sum(stats['completed']) / sum(stats['totals']) * 100), 1) 
                        if stats['totals'] and sum(stats['totals']) > 0 else 0.0
                    )
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting processing stats: {str(e)}", exc_info=True)
            return {
                'periods': [],
                'totals': [],
                'completed': [],
                'failed': [],
                'processing': [],
                'summary': {
                    'total': 0,
                    'completed': 0,
                    'failed': 0,
                    'processing': 0,
                    'success_rate': 0.0
                }
            }
    
    @staticmethod
    def delete_batch(batch_id: str, user_id: int) -> int:
        """
        Mark all results in a batch as deleted.
        
        Args:
            batch_id: The ID of the batch to delete
            user_id: The ID of the user (for security)
            
        Returns:
            Number of results deleted
        """
        with get_db_cursor(commit=True) as cur:
            cur.execute(
                "UPDATE results SET is_deleted = TRUE WHERE batch_id = %s AND user_id = %s",
                (batch_id, user_id)
            )
            return cur.rowcount
