import dataset_dg


def test_satellite_classification_dataset():
    data_dir = '/home/johnny/data_drive/digital_globe/test_classification_assets'
    dataset = dataset_dg.SatelliteClassificationDataset(data_dir)
    assert len(dataset) == 4
    for x in dataset:
        assert type(x) == tuple
        assert len(x) == 2
        image, label = x
        assert label == 0 or label == 1


def test_satellite_segmentation_dataset():
    data_dir = '/home/johnny/data_drive/digital_globe/test_segmentation_assets'
    dataset = dataset_dg.SatelliteSegmentationDataset(data_dir)
    assert len(dataset) == 2
    for x in dataset:
        assert type(x) == tuple
        assert len(x) == 2
        image, mask = x
        assert mask.size == image.size
        assert mask
