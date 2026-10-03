from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def convert_gps_to_decimal(gps_data):
    """
    Convert GPS EXIF coordinates into decimal latitude/longitude.
    """

    try:
        latitude = gps_data.get(GPSTAGS.get("GPSLatitude"))
        latitude_ref = gps_data.get(GPSTAGS.get("GPSLatitudeRef"))

        longitude = gps_data.get(GPSTAGS.get("GPSLongitude"))
        longitude_ref = gps_data.get(GPSTAGS.get("GPSLongitudeRef"))

        if not latitude or not longitude:
            return "Not available"

        def convert_to_decimal(coordinate):
            degrees = float(coordinate[0])
            minutes = float(coordinate[1])
            seconds = float(coordinate[2])

            return degrees + (minutes / 60) + (seconds / 3600)

        latitude_decimal = convert_to_decimal(latitude)
        longitude_decimal = convert_to_decimal(longitude)

        if latitude_ref == "S":
            latitude_decimal = -latitude_decimal

        if longitude_ref == "W":
            longitude_decimal = -longitude_decimal

        return {
            "latitude": round(latitude_decimal, 6),
            "longitude": round(longitude_decimal, 6)
        }

    except Exception:
        return "Not available"


def extract_image_metadata(image_path):
    """
    Extract available EXIF metadata from an image.

    Missing information is reported as 'Not available'.
    """

    metadata = {
        "camera_make": "Not available",
        "camera_model": "Not available",
        "date_taken": "Not available",
        "software": "Not available",
        "gps": "Not available"
    }

    try:
        image = Image.open(image_path)

        exif_data = image.getexif()

        if not exif_data:
            return metadata

        readable_exif = {}

        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            readable_exif[tag_name] = value

        metadata["camera_make"] = readable_exif.get(
            "Make",
            "Not available"
        )

        metadata["camera_model"] = readable_exif.get(
            "Model",
            "Not available"
        )

        metadata["date_taken"] = readable_exif.get(
            "DateTimeOriginal",
            readable_exif.get("DateTime", "Not available")
        )

        metadata["software"] = readable_exif.get(
            "Software",
            "Not available"
        )

        # Extract GPS information
        try:
            gps_info = image.getexif().get_ifd(34853)

            if gps_info:
                gps_data = {
                    GPSTAGS.get(tag_id, tag_id): value
                    for tag_id, value in gps_info.items()
                }

                metadata["gps"] = convert_gps_to_decimal(gps_data)

        except Exception:
            metadata["gps"] = "Not available"

        return metadata

    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return metadata