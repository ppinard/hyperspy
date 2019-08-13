# -*- coding: utf-8 -*-
# Copyright 2007-2019 The HyperSpy developers
#
# This file is part of HyperSpy.
#
# HyperSpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
#  HyperSpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HyperSpy.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path
from distutils.version import LooseVersion
from collections import namedtuple
from operator import attrgetter
import logging

import h5py
import numpy as np
import dask
import dask.array

from hyperspy.misc.utils import DictionaryTreeBrowser

_logger = logging.getLogger(__name__)

# Plugin characteristics
# ----------------------
format_name = 'Oxford Instruments NanoAnalysis'
description = 'Read H5OINA files, the HDF5 export format of Oxford Instruments NanoAnalysis.'
full_support = False
# Recognised file extension
file_extensions = ('h5oina',)
default_extension = 0
# Reading capabilities
reads_images = True
reads_spectrum = False
reads_spectrum_image = True
# Writing capabilities
writes = False
# ----------------------

# Additional constant variables
# ----------------------
MINIMUM_SUPPORTED_VERSION = LooseVersion('1.0')
DATASET_FORMAT_VERSION = 'Format Version'
DATASET_MANUFACTURER = 'Manufacturer'
DATASET_SOFTWARE_VERSION = 'Software Version'
DATASET_DETECTOR_TYPE_ID = 'Detector Type Id'
DATASET_BEAM_VOLTAGE = 'Beam Voltage'
DATASET_XCELLS = 'X Cells'
DATASET_XSTEP = 'X Step'
DATASET_YCELLS = 'Y Cells'
DATASET_YSTEP = 'Y Step'
TECHNIQUE_ELECTRON_IMAGE = 'Electron Image'
TECHNIQUE_EDS = 'EDS'
GROUP_HEADER = 'Header'
GROUP_DATA = 'Data'
GROUP_STAGE_POSITION = 'Stage Position'
GROUP_WINDOW_INTEGRAL = 'Window Integral'
GROUP_PEAK_AREA = 'Peak Area'

METADATA_LOOKUP_COMMON = [
    ('Project Notes', 'General.notes', None),
    ('Acquisition Date', 'General.date', lambda value: value.split('T')[0]),
    ('Acquisition Date', 'General.time', lambda value: value.split('T')[1]),
    ('Specimen Label', 'Sample.name', None),
    ('Beam Voltage', 'Acquisition_instrument.{instrument}.beam_energy', None),
    ('Working Distance', 'Acquisition_instrument.{instrument}.camera_length', None), # FIXME: Check units
    ('Magnification', 'Acquisition_instrument.{instrument}.magnification', None),
    ('Stage Position', 'Acquisition_instrument.{instrument}.stage.x', attrgetter('x_mm')), # FIXME: Check units
    ('Stage Position', 'Acquisition_instrument.{instrument}.stage.y', attrgetter('y_mm')), # FIXME: Check units
    ('Stage Position', 'Acquisition_instrument.{instrument}.stage.z', attrgetter('z_mm')), # FIXME: Check units
    ('Stage Position', 'Acquisition_instrument.{instrument}.stage.tilt_alpha', lambda value: np.degrees(value.tilt_rad)),
]

METADATA_LOOKUP_EDS = [
    ('Number Frames', 'Acquisition_instrument.{instrument}.Detector.EDS.number_of_frames', None),
    ('Detector Elevation', 'Acquisition_instrument.{instrument}.Detector.EDS.elevation_angle', np.degrees),
    ('Detector Azimuth', 'Acquisition_instrument.{instrument}.Detector.EDS.azimuth_angle', np.degrees),
    ('Detector Type Name', 'Acquisition_instrument.{instrument}.Detector.EDS.detector_type', None)
]

METADATA_LOOKUP_ELECTRON_IMAGE = [
    ('Dwell Time', 'Acquisition_instrument.{instrument}.dwell_time', None), # FIXME: Check units
]

StagePosition = namedtuple('StagePosition', ('x_mm', 'y_mm', 'z_mm', 'rotation_rad', 'tilt_rad'))

def file_reader(filename, *, lazy=True, select_type=None, **kwds):
    if Path(filename).suffix == '.h5oina':
        return read_h5oina(filename, lazy, select_type)
    return []

def read_h5oina(filename, lazy=True, select_type=None):
    with h5py.File(filename, mode='r') as f:
        _logger.debug(f'Reading {filename}')

        # Check version
        if DATASET_FORMAT_VERSION not in f:
            raise IOError('Invalid h5oina file: "Format Version" dataset is missing.')
        version = LooseVersion(f[DATASET_FORMAT_VERSION][0])

        if version < MINIMUM_SUPPORTED_VERSION:
            raise IOError(f'Format version ({version}) must be greater or equal to {MINIMUM_SUPPORTED_VERSION}')

        _logger.debug(f'Version: {version}')

        # As of version 1.0, only single acquisition is exported to h5oina.
        # Reading the Index dataset is skipped and we read the first slice
        # (aka frame in hyperspy nomenclature).
        index = '1'
        if index not in f:
            raise IOError('Invalid h5oina file: File does not contain any acquisition')

        slice_group = f[index]
        _logger.debug(f'Reading slice {index}')

        # Check techniques in slice.
        # H5OINA supports 3 techniques: EBSD, EDS and Electron Image.
        # Only EDS and Electron Image are supported in hyperspy.
        if TECHNIQUE_ELECTRON_IMAGE not in slice_group and TECHNIQUE_EDS not in slice_group:
            raise IOError(f'Invalid h5oina file: Slice {index} does not contain EDS nor Electron Image technique.')

        # Read different signals
        signal_dicts = []

        if select_type is None or select_type == 'image':
            signal_dicts += read_h5oina_electron_images(slice_group, lazy)
            signal_dicts += read_h5oina_eds_images(slice_group, lazy)

        if select_type == 'spectrum_image':
            raise IOError('Select type "spectrum_image" is not supported.')

    return signal_dicts

def read_h5oina_electron_images(slice_group, lazy=True):
    # Skip if no Electron Image technique group
    if TECHNIQUE_ELECTRON_IMAGE not in slice_group:
        return []

    technique_group = slice_group[TECHNIQUE_ELECTRON_IMAGE]
    _logger.debug('Reading electron images')

    # Initial checks
    if GROUP_HEADER not in technique_group:
        raise IOError(f'Invalid h5oina file: No Header group in technique {TECHNIQUE_ELECTRON_IMAGE}')
    if GROUP_DATA not in technique_group:
        raise IOError(f'Invalid h5oina file: No Data group in technique {TECHNIQUE_ELECTRON_IMAGE}')

    for key in [DATASET_XCELLS, DATASET_XSTEP, DATASET_YCELLS, DATASET_YSTEP]:
        if key not in technique_group[GROUP_HEADER]:
            raise IOError(f'Invalid h5oina file: Missing {key} in header')

    # Read header
    header_group = technique_group[GROUP_HEADER]
    header = read_h5oina_header(header_group)
    _logger.debug(header)

    # Extract hyperspy metadata from header
    lookup = METADATA_LOOKUP_COMMON + METADATA_LOOKUP_ELECTRON_IMAGE
    instrument = guess_instrument(header)
    filename = slice_group.file.filename
    signal_type = 'image'
    basemetadata = extract_metadata(lookup, header, instrument, filename, signal_type)

    # Define axes
    axes = [
        {
            'name': 'x',
            'offset': 0,
            'scale': header[DATASET_XSTEP],
            'size': header[DATASET_XCELLS],
            'units': 'µm',
            'navigate': False
        },
        {
            'name': 'y',
            'offset': 0,
            'scale': header[DATASET_YSTEP],
            'size': header[DATASET_YCELLS],
            'units': 'µm',
            'navigate': False
        },
    ]

    # Read images
    signal_dicts = []
    data_group = technique_group[GROUP_DATA]
    shape = (header[DATASET_YCELLS], header[DATASET_XCELLS])

    for detector_type in data_group:
        for name, dataset in data_group[detector_type].items():
            metadata = basemetadata.deepcopy()
            metadata.set_item('General.title', name)
            metadata.set_item(f'Acquisition_instrument.{instrument}.Detector.detector_type', detector_type)

            data = extract_data(dataset, shape, lazy)

            signal_dicts.append({
                'metadata': metadata.as_dictionary(),
                'original_metadata': header.copy(),
                'axes': axes,
                'data': data
            })

    return signal_dicts

def read_h5oina_eds_images(slice_group, lazy=True):
    # Skip if no EDS technique group
    if TECHNIQUE_EDS not in slice_group:
        return []

    technique_group = slice_group[TECHNIQUE_EDS]
    _logger.debug('Reading EDS images')
    _logger.debug(technique_group)

    # Initial checks
    if GROUP_HEADER not in technique_group:
        raise IOError(f'Invalid h5oina file: No Header group in technique {TECHNIQUE_EDS}')
    if GROUP_DATA not in technique_group:
        raise IOError(f'Invalid h5oina file: No Data group in technique {TECHNIQUE_EDS}')

    for key in [DATASET_XCELLS, DATASET_XSTEP, DATASET_YCELLS, DATASET_YSTEP]:
        if key not in technique_group[GROUP_HEADER]:
            raise IOError(f'Invalid h5oina file: Missing {key} in header')

    # Read header
    header_group = technique_group[GROUP_HEADER]
    header = read_h5oina_header(header_group)
    _logger.debug(header)

    # Extract hyperspy metadata from header
    lookup = METADATA_LOOKUP_COMMON + METADATA_LOOKUP_EDS
    instrument = guess_instrument(header)
    filename = slice_group.file.filename
    signal_type = 'image'
    basemetadata = extract_metadata(lookup, header, instrument, filename, signal_type)

    # Define axes
    axes = [
        {
            'name': 'x',
            'offset': 0,
            'scale': header[DATASET_XSTEP],
            'size': header[DATASET_XCELLS],
            'units': 'µm',
            'navigate': False
        },
        {
            'name': 'y',
            'offset': 0,
            'scale': header[DATASET_YSTEP],
            'size': header[DATASET_YCELLS],
            'units': 'µm',
            'navigate': False
        },
    ]

    # Read images
    signal_dicts = []
    data_group = technique_group[GROUP_DATA]
    shape = (header[DATASET_YCELLS], header[DATASET_XCELLS])

    for edstype in [GROUP_WINDOW_INTEGRAL, GROUP_PEAK_AREA]:
        if edstype not in data_group:
            continue

        for name, dataset in data_group[edstype].items():
            metadata = basemetadata.deepcopy()
            metadata.set_item('General.title', f'{edstype} - {name}')

            data = extract_data(dataset, shape, lazy)

            signal_dicts.append({
                'metadata': metadata.as_dictionary(),
                'original_metadata': header.copy(),
                'axes': axes,
                'data': data
            })

    return signal_dicts

def read_h5oina_header(header_group):
    """
    Read all datasets in the Header group of a technique.

    Returns
    -------
    :class:`dict`, where the keys are the dataset names and the values the dataset values
    """
    header = {}

    # Datasets in the root
    for key in [DATASET_SOFTWARE_VERSION, DATASET_MANUFACTURER, DATASET_FORMAT_VERSION]:
        if key in header_group.file:
            header[key] = header_group.file[key][0]

    # Datasets in the header group
    for key, value in header_group.items():
        if not isinstance(value, h5py.Dataset):
            continue

        value = value[()]
        if value.shape == (1,):
            value = value[0]
        else:
            value = tuple(value)

        header[key] = value

    # Special datasets
    if GROUP_STAGE_POSITION in header_group:
        stage_position = StagePosition(
            x_mm=header_group[GROUP_STAGE_POSITION].get('X')[0],
            y_mm=header_group[GROUP_STAGE_POSITION].get('Y')[0],
            z_mm=header_group[GROUP_STAGE_POSITION].get('Z')[0],
            rotation_rad=header_group[GROUP_STAGE_POSITION].get('Rotation')[0],
            tilt_rad=header_group[GROUP_STAGE_POSITION].get('Tilt')[0]
        )
        header[GROUP_STAGE_POSITION] = stage_position

    if DATASET_DETECTOR_TYPE_ID in header_group:
        header['Detector Type Name'] = header_group.get(DATASET_DETECTOR_TYPE_ID).attrs['Name']

    return header

def guess_instrument(header):
    """
    There is no way to tell whether the instrument is an SEM or TEM.
    This function guesses the instrument based on the beam voltage (TEM > 30kV).
    """
    beam_energy_keV = header.get(DATASET_BEAM_VOLTAGE, 0)
    return 'TEM' if beam_energy_keV > 30.0 else 'SEM'

def extract_metadata(lookup, header, instrument, filename, signal_type):
    """
    Extracts hyperspy metadata from header.

    Returns
    -------
    :class:`DictionaryTreeBrowser`
    """
    metadata = DictionaryTreeBrowser()

    # General metadata
    metadata.set_item('General.original_filename', filename)
    metadata.set_item('Signal.signal_type', signal_type)

    # Loop through the lookup
    for source_key, destination_key, transformer in lookup:
        if source_key not in header:
            continue

        destination_key = destination_key.format(instrument=instrument)

        value = header[source_key]
        if transformer is not None:
            value = transformer(value)

        metadata.set_item(destination_key, value)

    return metadata

def extract_data(dataset, shape, lazy):
    if lazy:
        return dask.array.from_array(dataset).reshape(shape)
    else:
        return np.asanyarray(dataset).reshape(shape)