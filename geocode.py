from geopy.geocoders import MapBox
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Geocoder:
    def __init__(self, access_token=None):
        # Use provided token or get from environment variables
        self.access_token = access_token or os.getenv("MAPBOX_ACCESS_TOKEN")
        if not self.access_token:
            print("Warning: MAPBOX_ACCESS_TOKEN not found in environment variables. Geocoding will fail.")
        self.geolocator = MapBox(api_key=self.access_token)

    def geocode(self, address):
        if not self.access_token:
            return None
        try:
            location = self.geolocator.geocode(address)
            return location
        except Exception as e:
            print(f"Geocoding error: {str(e)}")
            return None

    def reverse_geocode(self, coordinates):
        if not self.access_token:
            return None
        try:
            location = self.geolocator.reverse(coordinates)
            return location
        except Exception as e:
            print(f"Reverse geocoding error: {str(e)}")
            return None

# Create a Geocoder instance using environment variable
geocoder = Geocoder()

# Example usage (commented out):
# # Geocode an address (convert address to coordinates)
# location = geocoder.geocode("1600 Pennsylvania Ave NW, Washington, DC")
# if location:
#     print(f"Address: {location.address}")
#     print(f"Coordinates: ({location.latitude}, {location.longitude})")
# 
# # Reverse geocode (convert coordinates to address)
# location = geocoder.reverse_geocode("38.8977, -77.0365")
# if location:
#     print(f"Reverse geocoding result: {location.address}")