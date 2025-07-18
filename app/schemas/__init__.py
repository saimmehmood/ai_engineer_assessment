from app import ma
from marshmallow import fields, pre_load, validates_schema, ValidationError

# Default formats for date and datetime strings
DATE_FORMAT = '%Y-%m-%d'
DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%S'


class BaseSchema(ma.Schema):
    """
    A base schema that:
      - Trims whitespace on string values.
      - Removes keys with values that are None, empty, or the literal 'null'.
    """
    @pre_load
    def clean_input(self, data, **kwargs):
        cleaned = {}
        for key, value in data.items():
            # Trim string values
            if isinstance(value, str):
                value = value.strip()
            # Exclude empty values
            if value in [None, '', 'null']:
                continue
            cleaned[key] = value
        return cleaned


class CleanString(fields.String):
    """
    A custom string field that trims whitespace before deserialization.
    """
    def _deserialize(self, value, attr, data, **kwargs):
        if isinstance(value, str):
            value = value.strip()
        return super()._deserialize(value, attr, data, **kwargs)


class DateField(fields.Date):
    """
    A custom Date field with a default format.
    """
    def __init__(self, format=DATE_FORMAT, **kwargs):
        super().__init__(format=format, **kwargs)


class DateTimeField(fields.DateTime):
    """
    A custom DateTime field with a default format.
    """
    def __init__(self, format=DATETIME_FORMAT, **kwargs):
        super().__init__(format=format, **kwargs)


class DateRangeSchema(BaseSchema):
    """
    Schema for a date range with:
      - Optional 'date_from' and 'date_to' fields.
      - A pre_load step that normalizes date strings (removes a trailing 'T00:00:00' if present).
      - A validation to ensure that if both dates are provided, date_from ≤ date_to.
    """
    date_from = DateField(required=False)
    date_to = DateField(required=False)

    @pre_load
    def normalize_dates(self, data, **kwargs):
        for field in ['date_from', 'date_to']:
            if field in data and isinstance(data[field], str):
                # Remove unwanted time suffixes (e.g., "T00:00:00")
                data[field] = data[field].replace('T00:00:00', '')
        return data

    @validates_schema
    def validate_date_range(self, data, **kwargs):
        date_from = data.get('date_from')
        date_to = data.get('date_to')
        if date_from and date_to and date_from > date_to:
            raise ValidationError("date_from must be earlier than or equal to date_to.", field_names=['date_from', 'date_to'])


class DateTimeRangeSchema(BaseSchema):
    """
    Schema for a datetime range with similar behavior as DateRangeSchema.
    """
    datetime_from = DateTimeField(required=False)
    datetime_to = DateTimeField(required=False)

    @validates_schema
    def validate_datetime_range(self, data, **kwargs):
        datetime_from = data.get('datetime_from')
        datetime_to = data.get('datetime_to')
        if datetime_from and datetime_to and datetime_from > datetime_to:
            raise ValidationError("datetime_from must be earlier than or equal to datetime_to.",
                                  field_names=['datetime_from', 'datetime_to'])
