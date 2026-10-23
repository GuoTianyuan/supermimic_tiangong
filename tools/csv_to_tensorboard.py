"""Mirror a live training CSV to TensorBoard without changing the trainer."""
import argparse
import csv
import math
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    source = args.run_dir / 'train_log.csv'
    # Replaying the CSV on restart replaces the previous event stream from step 0.
    with SummaryWriter(str(args.run_dir / 'tensorboard'), purge_step=0) as writer:
        offset, fields = 0, None
        while True:
            if source.exists():
                with source.open() as stream:
                    stream.seek(offset)
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith('\n'):
                            stream.seek(start)
                            break
                        values = next(csv.reader([line]))
                        if fields is None:
                            fields = values
                        elif values != fields and len(values) == len(fields):
                            row = dict(zip(fields, values))
                            step = int(row['iteration'])
                            for key, value in row.items():
                                if key != 'iteration':
                                    value = float(value)
                                    if math.isfinite(value):
                                        writer.add_scalar('train/' + key, value, step)
                        offset = stream.tell()
                writer.flush()
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
