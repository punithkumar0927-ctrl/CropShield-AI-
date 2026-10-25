import numpy as np
from PIL import Image

from src.cropshield.image_quality import assess_image_quality, crop_largest_leaf_like_region


def test_dark_and_low_resolution_images_are_rejected_with_feedback():
    assessment = assess_image_quality(Image.new("RGB", (80, 100), color=(5, 5, 5)))

    assert not assessment.acceptable
    assert any("higher-resolution" in message for message in assessment.feedback)
    assert any("very dark" in message for message in assessment.feedback)


def test_sharp_well_lit_image_passes_basic_quality_check():
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[::2, ::2] = (35, 140, 40)
    pixels[1::2, 1::2] = (70, 185, 65)
    assessment = assess_image_quality(Image.fromarray(pixels))

    assert assessment.acceptable
    assert assessment.blur_score >= 20
    assert assessment.green_pixel_fraction > 0.4


def test_green_pixel_statistic_does_not_claim_to_detect_a_leaf():
    pixels = np.zeros((256, 256, 3), dtype=np.uint8)
    pixels[::2, ::2] = (30, 30, 180)
    pixels[1::2, 1::2] = (60, 60, 230)
    assessment = assess_image_quality(Image.fromarray(pixels))

    assert assessment.acceptable
    assert assessment.green_pixel_fraction == 0
    assert any("cannot identify a leaf" in message for message in assessment.feedback)


def test_quality_rejects_sub_224_and_blank_images():
    too_small = assess_image_quality(Image.new("RGB", (223, 300), color="green"))
    blank = assess_image_quality(Image.new("RGB", (300, 300), color="white"))

    assert not too_small.acceptable
    assert any("224 x 224" in message for message in too_small.feedback)
    assert not blank.acceptable
    assert any("blank or corrupted" in message for message in blank.feedback)


def test_leaf_like_segmentation_and_experimental_crop():
    pixels = np.full((300, 400, 3), 80, dtype=np.uint8)
    pixels[40:260, 70:330] = (35, 150, 45)
    image = Image.fromarray(pixels)

    assessment = assess_image_quality(image)
    crop = crop_largest_leaf_like_region(image)

    assert assessment.leaf_coverage_fraction > 0.3
    assert assessment.leaf_contour_count == 1
    assert crop is not None
    assert crop.width < image.width
    assert crop.height < image.height
