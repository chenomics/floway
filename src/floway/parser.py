"""FCS file parser."""

import struct
import pandas as pd
import re

class FCSParser:
    """
    A class to parse FCS 3.0/3.1 standard files from scratch.

    This parser reads the HEADER, TEXT, and DATA segments of an FCS file,
    extracting metadata and event data into a pandas DataFrame.
    """
    def __init__(self, filepath):
        """
        Initializes the parser with the path to the FCS file.

        Args:
            filepath (str): The path to the .fcs file.
        """
        self.filepath = filepath
        self.metadata = {}
        self.data = None
        self._read_file()

    def _read_header(self, f):
        """Reads the HEADER segment to get offsets for other segments."""
        header_str = f.read(58).decode('ascii')
        self.metadata['fcs_version'] = header_str[0:6].strip()

        # Extract byte offsets for TEXT, DATA, and ANALYSIS segments
        self.metadata['text_start'] = int(header_str[10:18].strip())
        self.metadata['text_end'] = int(header_str[18:26].strip())
        self.metadata['data_start'] = int(header_str[26:34].strip())
        self.metadata['data_end'] = int(header_str[34:42].strip())
        self.metadata['analysis_start'] = int(header_str[42:50].strip())
        self.metadata['analysis_end'] = int(header_str[50:58].strip())


    def _parse_text(self, text_segment_str):
        """Parses the key-value pairs in the TEXT segment."""
        # The first character is the delimiter
        delimiter = text_segment_str[0]

        # Check for escaped delimiters and handle them
        # A double delimiter means it's part of the value, not a separator
        if delimiter + delimiter in text_segment_str:
             # Use regex to split, looking for delimiters that are not followed by another delimiter
            # This is a robust way to handle escaped delimiters.
            kv_pairs = re.split(f'{re.escape(delimiter)}(?!{re.escape(delimiter)})', text_segment_str)
        else:
            kv_pairs = text_segment_str.split(delimiter)

        # The first element is empty because the string starts with the delimiter
        kv_pairs = kv_pairs[1:]

        # Create a dictionary from the key-value pairs
        i = 0
        while i < len(kv_pairs) - 1:
            key = kv_pairs[i].upper().strip()
            value = kv_pairs[i+1].strip()

            # Un-escape double delimiters in the value
            value = value.replace(delimiter+delimiter, delimiter)

            # Store key-value pair. Some keys can appear multiple times,
            # but we will overwrite with the last one for simplicity.
            self.metadata[key] = value
            i += 2

        # Standardize essential keywords
        self.metadata['total_events'] = int(self.metadata.get('$TOT', 0))
        self.metadata['num_parameters'] = int(self.metadata.get('$PAR', 0))

    def _read_data(self, f):
        """Reads the binary DATA segment and converts it to a numerical format."""
        num_events = self.metadata['total_events']
        num_params = self.metadata['num_parameters']

        # Determine byte order
        byte_order_str = self.metadata.get('$BYTEORD', '1,2,3,4')
        if byte_order_str == '1,2,3,4':
            byte_order = '<'  # Little-endian
        elif byte_order_str == '4,3,2,1':
            byte_order = '>'  # Big-endian
        else:
            # Default to native byte order if something else is specified
            byte_order = '='

        # Determine data type
        data_type = self.metadata.get('$DATATYPE', 'F')

        # Build the format string for the struct module
        struct_format = ''
        for i in range(1, num_params + 1):
            bits = int(self.metadata[f'$P{i}B'])
            if data_type == 'F':
                if bits == 32:
                    struct_format += 'f'
                else:
                    raise ValueError(f"Unsupported bit size for float: {bits}")
            elif data_type == 'I':
                if bits == 16:
                    struct_format += 'H' # Unsigned short
                elif bits == 32:
                    struct_format += 'I' # Unsigned int
                else:
                    raise ValueError(f"Unsupported bit size for int: {bits}")
            else:
                raise TypeError(f"Unsupported data type: {data_type}")

        full_struct_format = byte_order + struct_format

        # Seek to the start of the data segment
        f.seek(self.metadata['data_start'])

        # Read and unpack the data
        event_size = struct.calcsize(full_struct_format)
        raw_data = f.read(num_events * event_size)

        unpacked_data = []
        for i in range(num_events):
            event_bytes = raw_data[i * event_size : (i + 1) * event_size]
            unpacked_data.append(struct.unpack(full_struct_format, event_bytes))

        # Create a pandas DataFrame
        column_names = [self.metadata.get(f'$P{i}N', f'Param {i}') for i in range(1, num_params + 1)]
        self.data = pd.DataFrame(unpacked_data, columns=column_names)

    def _read_file(self):
        """Orchestrates the file reading process."""
        with open(self.filepath, 'rb') as f:
            self._read_header(f)

            # Read TEXT segment
            f.seek(self.metadata['text_start'])
            text_segment_bytes = f.read(self.metadata['text_end'] - self.metadata['text_start'] + 1)

            # Decode with error handling
            try:
                text_segment_str = text_segment_bytes.decode('utf-8')
            except UnicodeDecodeError:
                text_segment_str = text_segment_bytes.decode('latin-1', errors='ignore')

            self._parse_text(text_segment_str)

            # Read DATA segment only if it exists
            if self.metadata['data_end'] > 0 and self.metadata['total_events'] > 0:
                 self._read_data(f)
            else:
                 # Create an empty dataframe if there is no data
                 num_params = self.metadata.get('num_parameters', 0)
                 column_names = [self.metadata.get(f'$P{i}N', f'Param {i}') for i in range(1, num_params + 1)]
                 self.data = pd.DataFrame(columns=column_names)

    def get_data(self):
        """Returns the parsed data as a pandas DataFrame."""
        return self.data

    def get_metadata(self):
        """Returns the parsed metadata dictionary."""
        return self.metadata

    def list_channels(self):
        """
        Return a DataFrame describing each parameter (detector) in the file.

        Columns:
            index   - 1-based parameter index ($Pn).
            column  - $PnN, the detector short name (matches DataFrame column).
            stain   - $PnS, the operator-supplied stain/marker label (often blank).
            range   - $PnR, the detector range (max value).

        Use this to figure out which column corresponds to GFP/mCherry/etc.
        on a machine you haven't seen before.
        """
        rows = []
        for i in range(1, self.metadata.get('num_parameters', 0) + 1):
            rows.append({
                'index': i,
                'column': self.metadata.get(f'$P{i}N', f'Param {i}'),
                'stain': self.metadata.get(f'$P{i}S', ''),
                'range': self.metadata.get(f'$P{i}R', ''),
            })
        return pd.DataFrame(rows)
