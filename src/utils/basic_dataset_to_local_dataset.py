dataset_local_path = "D:\Data\Documents\AutomaticSaveToDisc\Datasets\ISIC-Archive-Downloader-master\Data"

with open("..\datasets\\test.dataset") as basic:
    print("opened basic dataset")
    basic_content = basic.readlines()
    basic_content = [x.strip() for x in basic_content]
    print("basic loaded")
    with open("..\datasets\\test_local_img.dataset", "w") as local_img:
        for fn in basic_content:
            local_img.write(dataset_local_path + "\Images\\" + fn + "_resized.jpg\n")
        print("local images done")
    with open("..\datasets\\test_local_json.dataset", "w") as local_json:
        for fn in basic_content:
            local_json.write(dataset_local_path + "\Descriptions\\" + fn + "\n")
        print("local json done")
