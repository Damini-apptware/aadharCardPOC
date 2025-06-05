import cv2
import pytesseract
import re

# Load image
image_path = "1.jpeg"
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)

aadhaar_pattern = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")
full_text = " ".join(data["text"])
matches = aadhaar_pattern.findall(full_text)

found = False
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.9
thickness = 2

if matches:
    for match in matches:
        groups = match.split()  # ['1234', '5678', '9012']
        if len(groups) == 3:
            masked_number = "XXXXXXXX "

            # Find all indices for each group
            group_indices_list = []
            for group in groups:
                indices_for_group = [i for i, text in enumerate(data["text"]) if text == group]
                group_indices_list.append(indices_for_group)

            # Number of occurrences based on first group appearances
            for occurrence_idx in range(len(group_indices_list[0])):
                # Get first group's bbox
                i1 = group_indices_list[0][occurrence_idx]
                x1, y1, w1, h1 = data["left"][i1], data["top"][i1], data["width"][i1], data["height"][i1]

                # Get second group's bbox (if exists)
                if occurrence_idx < len(group_indices_list[1]):
                    i2 = group_indices_list[1][occurrence_idx]
                    x2, y2, w2, h2 = data["left"][i2], data["top"][i2], data["width"][i2], data["height"][i2]
                else:
                    x2, y2, w2, h2 = x1, y1, w1, h1

                # Calculate combined bounding box to cover first 8 digits together
                x = min(x1, x2)
                y = min(y1, y2)
                w = (max(x1 + w1, x2 + w2)) - x
                h = max(h1, h2)

                # Mask the combined bbox with white rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), -1)

                # Get size of the text to be drawn
                (text_width, text_height), baseline = cv2.getTextSize(masked_number, font, font_scale, thickness)

                # Calculate position so text is centered
                text_x = x + (w - text_width) // 2
                text_y = y + (h + text_height) // 2  # y is top, so add height to go down

                # Put the masked text centered inside the white box
                cv2.putText(image, masked_number, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

            found = True

if found:
    cv2.imwrite("masked_output.jpg", image)
    print("✅ Masked image saved as 'masked_output.jpg'")
else:
    print("❌ Aadhaar number not found.")
