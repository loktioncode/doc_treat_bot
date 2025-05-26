from pydantic import BaseModel

class PatientVitals(BaseModel):
    heart_rate: float  # beats per minute
    blood_pressure_systolic: float  # mmHg
    blood_pressure_diastolic: float  # mmHg
    temperature: float  # degrees Celsius
    respiratory_rate: float  # breaths per minute
    oxygen_saturation: float  # percentage
    blood_glucose: float  # mg/dL

