# $Author: mhardin $
# $Date: 2013-06-25 00:27:28 -0400 (Tue, 25 Jun 2013) $
# $Revision: 730 $
# $HeadURL: file:///home/esp01/svn/code/libpython/trunk/irsa.py $
# $Id: irsa.py 730 2013-06-25 04:27:28Z mhardin $

# Updated Wed Feb 15 13:23:30 EST 2017
# Author: rchallen
# Fixed various issues with python package updates.

from __future__ import print_function
import astropy.io.fits as fits
import jd2date as j2d
import numpy as np
import datetime
import heapq
import os

'''
The routines in this file are designed to write data tables in the IPAC IRSA
format.

do_irsa:
       Generate headers, columns, and data for later use with 'irsa'
       and 'fits_writer'.
fits_writer:
       Uses lists and data arrays to produce a FITS table.
irsa:
       Uses lists and a data array to produce a table in the IPAC IRSA
       format.
'''

def do_irsa(event, fit, topstring = '', directory = '.',
            filename = None, set_aperr = True):
    """
    Generate headers, columns, and data for 'irsa' entries.
    
    Parameters:
    -----------
    event:  An event object
    fit:    A fit object
    topstring: String
            The top comment for the table. It should describe the data.
    filename: String
            The IRSA table will be saved to the (optional) filename.
    
    Raises:
    --------
    ValueError: Raises when 'topstring' is longer than 71 characters
                per line (maximum non-key characters allowed per line
                for FITS headers).

    Notes:
    ------
    - Attempts to read labels and units from event.fp.label or
      event.fp.unit. If it fails, then used hardcoded values.
    - Assumes all data are arranged in arrays of equal length.
    - Uses the 'event' object produced by our pipeline (along with a
      hard-coded list).

    Revisions:
    ----------
    2011-04-28  mhardin   Initial version             mhardin@knights.ucf.edu
    2011-05-05  mhardin   'columns' is a 2D list, header now handled more
                          efficiently, updated docstring, flipped data indices.
    2011-05-08  mhardin   Changed 'topstring', added bjdutc and bjdtdb,
                          correctly finds data from p11.
    2011-05-09  mhardin   Now includes data from 'fit' object.
    2011-05-18  mhardin   Changed 'data' to a list of 1D arrays, added data
                          types to 'columns'.
    2011-07-13  mhardin   Updated with fits_writer.
    2011-11-28  kevin     Added BLISS map.
    2013-09-24  patricio  Merged do_irsa and old_irsa.
                                                  pcubillos@fulbrightmail.org
    2014-04-13  patricio  Updated filename to not include the date of creation.
    2014-08-13  garland   Switched the pyfits package to astropy.io.fits.
			  zabblleon@gmail.com
    """

    # Dictionary for file naming scheme
    etype = {'s':'secondary','p':'primary','o':'orbit'}
    
    # Set up default file name
    if type(filename) == type(None):
        filename = directory + \
                   "/irsa/%s_%s_%s_Spitzer_%s_%.1f_microns"%(event.planetname,
                   etype[event.ecltype], j2d.jd2date(event.bjdutc[0,0]),
                   event.instrument.upper(), event.inst.spitzwavl*1e6)
    else:
        filename = directory + "/irsa/" + filename

    # Set aperture error to 0 to avoid format issues
    if set_aperr == True:
        event.fp.aperr[np.where(event.fp.good==0)] = 0.0
        
        
    # Make a list of the columns. Every name MUST correspond to an assigned
    # value in 'event.fp', 'event', or 'fit' (in that order):
                  # Name         Description                        Data type
    columns = [['frmobs',    'sequential frame number',             'int'  ],
               ['pos',       'position number',                     'int'  ],
               ['aor',       'sequential AOR number',               'int'  ],
               ['expid',     'EXPosure ID',                         'int'  ],
               ['dce',       'Data Collection Event',               'int'  ],
               ['subarn',    'subarray frame number',               'int'  ],
               ['cycpos',    'cycle number',                        'int'  ],
               ['visobs',    'visit number within observation set', 'int'  ],
               ['im',        'frame within position',               'int'  ],
               ['frmvis',    'frame within visit',                  'int'  ],
               ['x',         'X position',                          'float'],
               ['y',         'Y position',                          'float'],
               ['r',         'distance from nearest pixel center',  'float'],
               ['good',      'good flag',                           'int'  ],
               ['clipmask',  'points used in model fit',            'int'  ],
               ['aplev',     'aperture flux',                       'float'],
               ['aperr',     'aperture error',                      'float'],
               ['apraw',     'raw photometry (no sky subtraction)', 'float'],
               ['nappix',    'number of aperture pixels',           'float'],
               ['skylev',    'sky level',                           'float'],
               ['skyerr',    'sky error',                           'float'],
               ['medsky',    'median sky',                          'float'],
               ['nskypix',   'number of sky pixels',                'float'],
               ['nskyideal', 'ideal number of sky pixels',          'float'],
               ['time',      'frame mid-time, seconds J2000.0',     'float'],
               ['bjdcor',    'light-time correction to BJD',        'float'],
               ['bjdutc',    'barycentric universal time',          'float'],
               ['bjdtdb',    'barycentric dynamical time',          'float'],
               ['phase',     'orbital phase',                       'float'],
               ['bestfit',   'best-fit model applied to data',      'float'],
               ['bestmip',   'best-fit BLISS map',                  'float'],
              ]

    # Constants
    maxlen  = 71    # max length of BLANK string in FITS files

    # Edit ORIGIN in event.header:
    event.header['ORIGIN'] = ('University of Central Florida Planetary Science',
                              'file creator')

    # Remove SIMPLE, BITPIX, NAXIS from event.header:
    dellist = ['SIMPLE',
               'BITPIX',
               'NAXIS',
               'NAXIS1',
               'NAXIS2',
               'NAXIS3']
    
    for item in dellist:
        try:
            del event.header[item]
        except:
            print('Keyword ' + item + ' not found. Skipping deletion.')
    

    # Split 'topstring' into a list of lines and add to event.header
    toplist = topstring.split('\n')

    # Remove white space to the left of each line
    # Check each line to make sure it is less than 71 characters
    for line in range(len(toplist)):
        toplist[line] = toplist[line].lstrip(' ')
        if len(toplist[line]) > maxlen:
            raise ValueError(
            "Line " + str(line) + " in 'top' is longer than 71 characters")
        
    # Check to make sure that 'topstring' is not already present
    if topstring.startswith(str(event.header[1]).lstrip(' ')):
            print("\'topstring\' already present, not duplicating")
            
    # Write to event.header if it passes all checks
    else:
        event.header.add_blank('', before = 0)
        for line in range(len(toplist)):
            event.header.add_blank(toplist[line], after = line)
        event.header.add_blank('', before = 'ORIGIN')

    # Make event.header into a list of strings, 'header', for the IRSA table
    header = str(event.header).splitlines()
    header.append('')

    # Find the longest name in 'columns'
    longest = len(columns[0][0])
    for line in range(len(columns)):
        if len(columns[line][0]) > longest:
            longest = len(columns[line][0])

    # Make the second part of the header, 'key', from 'columns':
    key  = []
    desc = []
    for line in range(len(columns)):
        name = columns[line][0].upper()
        desc.append(columns[line][1])
        if ( (name == 'X') or (name == 'Y') or (name == 'Z') ):
            name = name + '_POS'
        key.append('{0:{width}}'.format(name, width=longest + 2) + desc[line])


    # Get the length of the flattened data from 'event.fp.param'
    # where 'param' is the first entry in 'columns'
    datalen = len(event.fp.aplev.flatten())

    # Get the length of the unclipped and clipped model fits
    fitdatalenuc = len(fit.fluxuc)
    fitdatalen   = len(fit.flux)

    # Make 'order' to sort the data by time
    try:
        order = np.argsort(event.bjdutc, axis = None)
    except:
        print('No BJD_UTC values found. Data may not be sorted properly.')
    
    # Get the data from 'event.fp.param'
    data = []
    remove = []
    for item in range(len(columns)):
        found = False
        for attr, value in event.fp.__dict__.iteritems():
            if columns[item][0] == attr:
                found = True
                if   columns[item][2] == 'int':
                    data.append( np.array(value.flat[order], dtype='int64') )
                elif columns[item][2] == 'float':
                    data.append( np.array(value.flat[order], dtype='float64') )
        # Check 'event.param' if data not found:
        if not found:
            for attr, value in event.__dict__.iteritems():
                if columns[item][0] == attr:
                    found = True
                    if   columns[item][2] == 'int':
                        data.append(np.array(value.flat[order], dtype='int64'))
                    elif columns[item][2] == 'float':
                        data.append(np.array(value.flat[order], dtype='float64'))
        # Check 'fit.param' if data still not found:
        if not found:
            for attr, value in fit.__dict__.iteritems():
                if columns[item][0] == attr:
                    found = True
                    if len(value) == fitdatalen:
                        foouc = np.zeros(fitdatalenuc) + np.nan
                        foouc[np.where(fit.clipmask)] = value
                    else:
                        foouc = value
                    foo = np.zeros(datalen)
                    foo[np.where(event.good.flat)] = foouc
                    if columns[item][2] == 'int':
                        #foo[np.where(event.good.flat == 0)] = 0
                        data.append(np.array(foo[order], dtype='int64'))
                    elif columns[item][2] == 'float':
                        data.append(np.array(foo[order], dtype='float64'))
        if not found:
            print("Item '%s' not found in data."%columns[item][0])
            remove.append(item)
    for item in remove:
      dummy = columns.pop(item)

    # Get the column names and units:
    colnames = []
    units    = []
    try: # From 'event.fp.label.param' and 'event.fp.unit.param'
        for item in range(len(columns)):
            for attr, value in event.fp.label.__dict__.iteritems():
                if columns[item][0] == attr:
                    colnames.append(value)
            for attr, value in event.fp.unit.__dict__.iteritems():
                if columns[item][0] == attr:
                    units.append(value)
    except: # From 'columns'
        for item in range(len(columns)):
            value = columns[item][0]
            if ( value in ['x', 'y', 'z'] ):
                colnames.append(value.upper() + '_POS')
                print('Changed', value.upper(), 'to', value.upper() + '_POS')
            else:
                colnames.append(value.upper())

            # The units
            if   value in ['time', 'bjdcor']:
                units.append('seconds_J2000')
            elif value in ['bjdutc', 'bjdtdb']:
                units.append('days')
            elif value == ['medsky', 'skylev', 'skyerr']:
                units.append('microJy_per_pix')
            elif value in ['aplev', 'aperr', 'apraw', 'bestfit']:
                units.append('microJy')
            elif value in ['x', 'y', 'z']:
                units.append('pixels')
            else:
                units.append('')

    # Call fits_writer:
    print('Writing FITS table')
    fits_writer(data, event.header, colnames, desc, filename)

    # Call irsa:
    print('Writing IRSA table')
    irsa(data, header, key, colnames, units, filename)

    return

def fits_writer(data, fitsheader, colnames, desc,
                filename="evtname.fits", binary = False):
    '''
    This function takes in several data arrays and lists of strings to produce
    a FITS data table. The table is saved as 'filename'.
    
    Parameters:
    -----------
    data:         list
                  A list of numpy 1D array objects.
    fitsheader:   PyFITS .header object
                  Header to be included in the FITS file.
    colnames:     string list
                  Each entry is one column name.
    desc:         string list
                  Each entry is the description for a column.
    filename:     string
                  The FITS table will be saved to the (optional) filename.
    binary:       boolean
                  Determines type of table. Choose True for binary, False for
                  ascii. Default is False.
    
    Returns:
    -------
    None
        
    Notes:
    -----
    All string lists MUST be lists, even if only containing one element.

    'fitsheader' should be fitsfile.header when the file 'fitsfile.fits' is
    read into Python.

    Example:
    -------
    >>> import numpy as np
    >>> import irsa
    >>>
    >>> data     = [ np.arange(10), np.arange(10.) ]
    >>> header   = fitsfile.header
    >>> colnames = ['col1','col2']
    >>> desc     = ['first column', 'second column']
    >>> irsa.fits_writer(data, header, colnames, desc)
    >>>
    
    Revisions:
    ----------
    2011-07-13: mhardin@knights.ucf.edu  initial version
    2012-12-18: mhardin@knights.ucf.edu  changed width calculation method to
                                         stop underestimating the width
    '''
    
    # Define constants
    allnum  = '0123456789'  # integers to be stripped from string
    padding = 3             # number of extra spaces in each column
#    declen  = 10             # max number of decimal places
#    intform    = 'I6'          # integer column format
#    floatform  = 'F18.7'      # float column format
#    doubleform = 'D18.7'      # double column format
#    charform   = 'S16'         # character column format

    filename = filename + '.fits'

    # Split each value into a whole number and a decimal, then get the length
    # of each. Store the largest whole number and decimal lengths per column.
    # Finally, combine largest whole number and decimal lengths to get a column
    # width. Keep column width and the decimal length for use in FITS
    # formatting code.
    maxlen = np.zeros( len(data), dtype='int64' )     # Maximum column length
    maxdec = np.zeros( len(data), dtype='int64' )     # Maximum decimal length
    maxpredec  = np.zeros( len(data), dtype='int64' ) # Largest whole number per column
    maxpostdec = np.zeros( len(data), dtype='int64' ) # Largest decimal per column
    for column in range(len(data)):
        for row in range(len(data[column])):
            # Check if value is a float
            # print(data[column][row])
            if str(data[column].dtype).rstrip(allnum) == 'float':
                # Check if float is a nan/inf.
                # Can't split by '.' if it is a nan/inf!
                if np.isnan(data[column][row]):
                    predec  = len('nan')
                    postdec = 0
                elif np.isinf(data[column][row]):
                    predec  = len('inf')
                    postdec = 0
                else:
                    predec  = len( str(data[column][row]).split('.')[0] ) + 1
                    postdec = len( str(data[column][row]).split('.')[1] )
            else:
                predec  = len( str(data[column][row]) )
                postdec = 0
            # Store the largest whole number and decimal number lengths per
            if maxpredec[column] < predec:                           # column.
                maxpredec[column] = predec
            if maxpostdec[column] < postdec:
                maxpostdec[column] = postdec
        # Add the two lengths together to get a column width.
        maxlen[column] = maxpredec[column] + maxpostdec[column]
        maxdec[column] = maxpostdec[column]  # Keep the decimal width for later

    # Load data into list of column objects
    fitscol = []
    for column in range(len(data)):
        colname = colnames[column]
        coldata = data[column]
        # Get number of decimal places. If zero, do not include in 'length'.
        if maxdec[column] == 0:
            length = str(maxlen[column] + padding)
        else:
            length = str(maxlen[column] + padding) + '.' + str(maxdec[column])
        # Determine the data type of the column
        if str(coldata.dtype).rstrip(allnum) == 'int':
            coltype = 'I' + length
        elif str(coldata.dtype).rstrip(allnum) == 'float' and binary == False:
            coltype = 'F' + length
        elif str(coldata.dtype).rstrip(allnum) == 'float' and binary == True:
            coltype = 'E' + length
        elif str(coldata.dtype).rstrip(allnum) == '|S':
            coltype = 'A' + length
        else:
            coltype = 'F' + length
        col = fits.Column(name=colname, format=coltype, array=coldata)
        fitscol.append(col)
    
    # Generate a table HDU object
    if binary == False:
        tbhdu = fits.TableHDU.from_columns(fitscol)
    else:
        tbhdu = fits.BinTableHDU.from_columns(fitscol)

    # Add comments to the table header
    tbheader = tbhdu.header
    for column in range(len(data)):
        tbnum = column + 1   # the first column is 1, not 0
        ttype = 'TTYPE' + str(tbnum)
        # Update existing 'TTYPE' key with comment from 'desc'
        tbheader[ttype] = (tbheader[ttype], desc[column])

    # Make FITS file using 'fitsheader' and 'tbhdu'
    hdu = fits.PrimaryHDU(None, fitsheader)
    tbhdulist = fits.HDUList([hdu, tbhdu])
#    try:
    tbhdulist.writeto(filename)
#    except:
#        print('Format error occurred while writing file. Attempting to fix.')
#        tbhdulist[0].verify('fix')
#        tbhdulist[1].verify('fix')
#        tbhdulist.writeto(filename)

    return

def irsa(data, header, key, colnames,
         units, filename = "evtname_irsa.tbl", null='nan'):
    '''
    This function takes in a data array and lists of strings to produce an IPAC
    IRSA table. The table is saved as 'filename'.
    
    Parameters:
    -----------
    data:     list
              A list of numpy 1D array objects.
    header:   string list
              Each entry is one line in the table's header.
    key:      string list
              Each entry is one line describing the column names.
    colnames: string list
              Each entry is one column name.
    units:    string list
              Each entry is one column's units.
    filename: string
              The IRSA table will be saved to the (optional) filename.
    comment:  string list
              Each entry is one line after the table's header. Optional.
    null:     string
              A keyword indicating the null value in the data set.
              Default is 'nan'.
    
    Returns:
    -------
    None
    
    Raises:
    ------
    ValueError: Raises when the number of columns/units do not match the number
                of columns in the data array.
    
    Notes:
    -----
    All string lists MUST be lists, even if only containing one element.
    
    Example:
    -------
    >>> import numpy as np
    >>> import irsa
    >>>
    >>> data     = [ np.arange(10), np.arange(10.) ]
    >>> header   = ['This is a test', 'what is the length?']
    >>> key      = ['col1  a column', 'col2  another column']
    >>> colnames = ['col1','col2']
    >>> units    = ['mJy', 'badpix']
    >>> irsa.irsa(data, header, key, colnames, units)
    >>>
    
    Revisions:
    ----------
    2011-04-28  mhardin@knights.ucf.edu  initial version
    2011-05-05  mhardin@knights.ucf.edu  added header formatting, 'keys' param,
                                         updated docstring, flipped data indices
    2011-05-18  mhardin@knights.ucf.edu  changed 'data' to a list of 1D arrays
    '''
    
    # Constants
    allnum   = '0123456789'    # used for stripping numbers from a string
    
    # Edit 'filename' to include time of creation and extension
    filename = filename + '.tbl'
    
    # Raise error if 'data' and 'colnames' or 'units' do not have the same number
    # of columns
    if ( len(data) != len(colnames) ) or ( len(data) != len(units) ):
        raise ValueError("Number of columns (%i) data (%i) and/or units (%i) "
                         "entries do not match!"%(len(colnames), len(data),
                                                  len(units)))

    # Edit the header to meet IPAC standards
    # Add '\ ' or '\' for header comments and keywords (respectively)
    for line in range(len(header)):
        if len(header[line].lstrip('\ ')) > 8:
            is_keyword    = (header[line].lstrip('\ ')[8] == '=')
            # Need to prepend '\' if keyword
            if header[line].lstrip('\ ') and is_keyword:
                header[line] = '\\' + header[line].lstrip('\ ')
            else:
                header[line] = '\ ' + header[line].lstrip('\ ')
        elif header[line].startswith('\ '):
            pass
        elif header[line].startswith('\\') or header[line].startswith(' '):
            header[line] = '\ ' + header[line].lstrip('\ ')
        else:
            header[line] = '\ ' + header[line]
        # Remove extra whitespace
        header[line] = " ".join(header[line].split())
    
    # Append 'key' and add '\ ' to each line
    for line in range(len(key)):
        key[line] = '\ ' + key[line].lstrip('\ ')
        header.append(key[line])
    
    # Write header at the top of the IRSA table
    with open(filename, 'w') as f:
        for string in range(len(header)):
            print(header[string], file=f)
    
    # Find widths of all values in a column, get largest, and store in 'maxl'
    maxl = np.zeros( len(data) )
    for column in range(len(data)):
        for row in range(len(data[column])):
            namel = len( str(colnames[column]) )
            typel = len( str(data[column][row].dtype).rstrip(allnum) )
            unitl = len( str(units[   column]) )
            datal = len( str(data[column][row]) )
            sizes = (namel, typel, unitl, datal)
            largest = heapq.nlargest(1, sizes)[0]
            if maxl[column] < largest:
                maxl[column] = largest
    
    # Write column names
    with open(filename, "a") as f:
        print('|', end = '', file = f)
        for column in range(len(data)):
            print(' {0:{width}}'.format(colnames[column],
                           width = np.int(maxl[column])), end = ' |', file = f)
        print('', file = f)
    	
    # Write data type
    with open(filename, "a") as f:
        print('|', end = '', file = f)
        for column in range(len(data)):
            dtype = str(data[column][0].dtype).rstrip(allnum)
            print(' {0:{width}}'.format(dtype,
                           width = np.int(maxl[column])), end = ' |', file = f)
        print('', file = f)
    
    # Write units
    with open(filename, "a") as f:
        print('|', end = '', file = f)
        for column in range(len(data)):
            print(' {0:{width}}'.format(units[column],
                           width = np.int(maxl[column])), end = ' |', file = f)
        print('', file = f)
    
    # Write null value
    with open(filename, "a") as f:
        print('|', end = '', file = f)
        for column in range(len(data)):
            print(' {0:{width}}'.format(null,
                            width = np.int(maxl[column])), end = ' |', file = f)
        print('', file = f)
    
    # Write the data
    with open(filename, "a") as f:
        for row in range(len(data[0])):
            print(' ', end = '', file = f)
            for column in range(len(data)):
                print(' {0:{width}}'.format(str(data[column][row]),
                            width = np.int(maxl[column])), end = '  ', file = f)
            print('', file = f)
    
    return

"""
def fits_writer(data, fitsheader, colnames, desc,
                filename="evtname.fits", binary = False):
    '''
    This function takes in several data arrays and lists of strings to produce
    a FITS data table. The table is saved as 'filename'.
    
    Parameters:
    -----------
    data:         list
                  A list of numpy 1D array objects.
    fitsheader:   PyFITS .header object
                  Header to be included in the FITS file.
    colnames:     string list
                  Each entry is one column name.
    desc:         string list
                  Each entry is the description for a column.
    filename:     string
                  The FITS table will be saved to the (optional) filename.
    binary:       boolean
                  Determines type of table. Choose True for binary, False for
                  ascii. Default is False.
    
    Returns:
    -------
    None
        
    Notes:
    -----
    All string lists MUST be lists, even if only containing one element.

    'fitsheader' should be fitsfile.header when the file 'fitsfile.fits' is
    read into Python.

    Example:
    -------
    >>> import numpy as np
    >>> import irsa
    >>>
    >>> data     = [ np.arange(10), np.arange(10.) ]
    >>> header   = fitsfile.header
    >>> colnames = ['col1','col2']
    >>> desc     = ['first column', 'second column']
    >>> irsa.fits_writer(data, header, colnames, desc)
    >>>
    
    Revisions:
    ----------
    2011-07-13: mhardin@knights.ucf.edu  initial version
    '''
    
    # Define constants
    allnum  = '0123456789'  # integers to be stripped from string
    padding = 3             # number of extra spaces in each column
    numdec  = 10             # max number of decimal places
#    intform    = 'I6'          # integer column format
#    floatform  = 'F18.7'      # float column format
#    doubleform = 'D18.7'      # double column format
#    charform   = 'S16'         # character column format


    # Edit 'filename' to include time of creation
    filename = filename + \
               datetime.datetime.strftime(datetime.datetime.today(),
               "_%Y_%m_%d_%H_%M") + '.fits'

    # Find widths of all values in a column, get largest, and store in 'maxl'
    # Also find number of decimal places (if any) and store in 'decl'
    maxl = np.zeros( len(data), dtype='int64' )
    decl = np.zeros( len(data), dtype='int64' )
    for column in range(len(data)):
        for row in range(len(data[column])):
            datal = len( str(data[column][row]) )
            #if colnames[column] == 'NSKYPIX' and row == 1000:
            #    print(colnames[column], ':', repr(data[column][1000]))
            #    print('Old version:', str(data[column][1000]))
            #    print(data[column][1000])
            try:
                dec = len( str(data[column][row]).split('.')[1] )
            except:
                dec = 0
            if maxl[column] < datal:
            #    print('maxl =', maxl[column])
            #    print(colnames[column], data[column][row], datal)
                maxl[column] = datal
            if decl[column] < dec:
                decl[column] = dec

    # Load data into list of column objects
    fitscol = []
    for column in range(len(data)):
        colname = colnames[column]
        coldata = data[column]
        # Get number of decimal places. If zero, do not include in 'length'.
        if decl[column] == 0:
            length = str(maxl[column] + padding)
        elif decl[column] <= numdec:
            length = str(maxl[column] + padding) + '.' + str(decl[column])
        else:
            length = str(maxl[column] + padding) + '.' + str(numdec)
        print(colname, 'maxl:', maxl[column], 'decl:', decl[column], 'len:', length)
        # Determine the data type of the column
        if str(coldata.dtype).rstrip(allnum) == 'int':
            coltype = 'I' + length
        elif str(coldata.dtype).rstrip(allnum) == 'float' and binary == False:
            coltype = 'F' + length
        elif str(coldata.dtype).rstrip(allnum) == 'float' and binary == True:
            coltype = 'E' + length
        elif str(coldata.dtype).rstrip(allnum) == '|S':
            coltype = 'A' + length
        else:
            coltype = 'F' + length            
        col = fits.Column(name=colname, format=coltype, array=coldata)
        fitscol.append(col)
    
    # Generate a table HDU object
    if binary == False:
        tbhdu = fits.new_table(fitscol, tbtype = 'TableHDU')
    else:
        tbhdu = fits.new_table(fitscol, tbtype = 'BinTableHDU')

    # Add comments to the table header
    tbheader = tbhdu.header
    for column in range(len(data)):
        tbnum = column + 1   # the first column is 1, not 0
        ttype = 'TTYPE' + str(tbnum)
        # Update existing 'TTYPE' key with comment from 'desc'
        tbheader.update(ttype, tbheader[ttype], desc[column])

    # Make FITS file using 'fitsheader' and 'tbhdu'
    hdu = fits.PrimaryHDU(None, fitsheader)
    tbhdulist = fits.HDUList([hdu, tbhdu])
#    try:
    tbhdulist.writeto(filename)
#    except:
##        print('Format error occurred while writing file. Attempting to fix.')
#        tbhdulist[0].verify('fix')
#        tbhdulist[1].verify('fix')
#        tbhdulist.writeto(filename)

    return
"""
