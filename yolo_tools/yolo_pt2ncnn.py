import argparse


def main():
    parser = argparse.ArgumentParser(description="Convert .pt to ncnn format for YOLOv8 models.")
    parser.add_argument(
        "path_to_pt", type=str
    )
    args = parser.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.path_to_pt)
    model.export(format='ncnn')

if __name__ == "__main__":
    main()
