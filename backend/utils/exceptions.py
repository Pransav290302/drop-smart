"""Custom exceptions for DropSmart"""


class DropSmartException(Exception):
    """Base exception for DropSmart"""
    pass


class FileValidationError(DropSmartException):
    """Raised when file validation fails"""
    pass


class SchemaValidationError(DropSmartException):
    """Raised when Excel schema validation fails"""
    pass


class ProcessingError(DropSmartException):
    """Raised when data processing fails"""
    pass


class ModelNotFoundError(DropSmartException):
    """Raised when ML model is not found"""
    pass


class ModelPredictionError(DropSmartException):
    """Raised when model prediction fails"""
    pass

