import numpy as np
from numba import jit

@jit(nopython=True)
def sign(a):
    '''Returns the sign of float'''
    if a < 0:
        sign = -1
    elif a > 0:
        sign = +1
    elif a == 0:
        sign = 0
    return sign

@jit(nopython=True)
def select_line(ref, wvl, flux):
    ## Line selection
    lim = 400 # km/s  We define a limit to not exceed
    d = (ref - wvl)**2
    i = np.where(d==np.min(d))[0][0] # position of closest point to line
    deriv = np.diff(flux) # Flux derivative

    ## Now we seach for when the derivative changes sign
    j = 1
    shift = (wvl[i+j]-wvl[i])/wvl[i]*(3*1e5) # We keep track of the shift
    deriv_sign = sign(deriv[i+j+1])
    while shift<lim: # Do not go too far
        if sign(deriv[i+j]) != deriv_sign:
            lb = shift
            break
        else:
            shift = (wvl[i+j]-wvl[i])/wvl[i]*(3*1e5)
            lb = shift
        j+=1
    j = -1
    shift = (wvl[i]-wvl[i+j])/wvl[i]*(3*1e5)
    deriv_sign = sign(deriv[i+j-1])
    while shift<lim:
        if sign(deriv[i+j]) != deriv_sign:
            hb = shift
            break
        else:
            shift = (wvl[i]-wvl[i+j])/wvl[i]*(3*1e5)
        j-=1
    hws = (hb+lb)/2
    return hws

@jit(nopython=True)
def select_lines(lines, wvl, flux):
    hwss = np.empty(lines.shape)
    for i in range(len(lines)):
        if (lines[i]<wvl[0]) | (lines[i]>wvl[-1]):
            hws = 0
        else:
            hws = select_line(lines[i], wvl, flux)
        hwss[i] = hws
    return hwss

def gen_bounds(wlines, hwss):
    bounds = []
    for i in range(len(wlines)):
        lim = hwss[i]*wlines[i]/(3*1e5)
        bounds.append([wlines[i]-lim, wlines[i]+lim])
    return bounds

def filter(wlines, bounds):
    nbounds = []
    nwlines = []
    idxtoremove = []
    for i in range(len(bounds)):
        if (bounds[i][1]-bounds[i][0])==0:
            idxtoremove.append(i)
            continue
        else:
            nbounds.append(bounds[i])
            nwlines.append(wlines[i])
    return nwlines, nbounds, idxtoremove

def increase_width(bounds, pad=0.05):
    nbounds = []
    for i in range(len(bounds)):
        lb, hb = bounds[i]
        width = hb - lb
        _lb = lb - pad * width
        _hb = hb + pad * width
        nbounds.append([_lb, _hb])
    return nbounds

def merge(bounds):
    '''Merge regions that are overlapping'''
    nbounds = []
    i=0
    while i < (len(bounds)): ## Loop through all the bounds in list
        if (bounds[i][1] - bounds[i][0])==0: ## This is an empty bound
            i+=1
            continue 
        _nbound = bounds[i].copy()
        j=1
        cond = False
        while cond is False:
            if (i+j) >= len(bounds): ## we've gone through all of them
                nbounds.append(_nbound)
                i = len(bounds) # exit function
                cond = True
            elif bounds[i+j][0] < _nbound[1]: # Overlap definition
                _nbound[1] = max([_nbound[1], bounds[i+j][1]])
                _nbound[0] = min([_nbound[0], bounds[i+j][0]])
                j += 1
            else:
                nbounds.append(_nbound)
                i = i + j
                cond = True
    if len(bounds)==1:
        nbounds = bounds
    return nbounds

def make_regions(wvl, flux, bounds, mask, length=400):
    '''
    Make regions from all the information on the lines and regions
    '''
    edges = int(1/8*length)
    length -= 2*edges
    # currentorder = orders[0]
    # order = orders[0]
    # while order==currentorder:
    lb, hb = bounds[0]
    if wvl[0]>=lb: # If the bounds were wider than the region
        start=0
    else:
        start = np.where(wvl[wvl<lb])[0][-1] # Start as close as possible to bound
    end = start + length # End *length* bins later
    wvl_regions = []
    flux_regions = []
    nmasks = []
    i = 0
    while i < len(bounds):
        lb, hb = bounds[i]
        ## If window was too short to account for the span, we split window and
        ## start over
        if end+edges > len(wvl):
            ## If out of bounds, change the start of the window 
            i+=1 # We simply pass to the next iteration
        elif (hb > wvl[end]) & (i==(len(bounds)-1)):
            _nmask = np.zeros(length+2*edges)
            for k in range(len(mask)):
                if (wvl[start]<mask[k][0]) & (wvl[end]>mask[k][1]):
                    _nmask[(wvl[start-edges:end+edges]>mask[k][0]) 
                            & (wvl[start-edges:end+edges]<mask[k][1])] = 1
                ## If the mask englobs the entirety of the window
                if (wvl[start]>=mask[k][0]) & (wvl[end]<=mask[k][1]):
                    _nmask[:] = 1
            nmasks.append(_nmask)
            wvl_regions.append(wvl[start-edges:end+edges])
            flux_regions.append(flux[start-edges:end+edges])
            start = end - edges # Safeguards lines ingnored because of the edges
            end = start+length
        ## If we reached the final window, we need not extend boundaries
        elif (i==(len(bounds)-1)):
            _nmask = np.zeros(length+2*edges)
            for k in range(len(mask)):
                if (wvl[start]<mask[k][0]) & (wvl[end]>mask[k][1]):
                    _nmask[(wvl[start-edges:end+edges]>mask[k][0]) 
                            & (wvl[start-edges:end+edges]<mask[k][1])] = 1
            nmasks.append(_nmask)
            wvl_regions.append(wvl[start-edges:end+edges])
            flux_regions.append(flux[start-edges:end+edges])
            i+=1
        ## If the hb is in the window we look for the next window
        elif hb < wvl[end]: # If hb is in the window
            i+=1
        ## If hb out of window, we chunck a window and start a new one
        else:
            _nmask = np.zeros(length+2*edges)
            for k in range(len(mask)):
                if (wvl[start]<mask[k][0]) & (wvl[end]>mask[k][1]):
                    _nmask[(wvl[start-edges:end+edges]>mask[k][0]) 
                            & (wvl[start-edges:end+edges]<mask[k][1])] = 1
            nmasks.append(_nmask)
            wvl_regions.append(wvl[start-edges:end+edges])
            flux_regions.append(flux[start-edges:end+edges])
            start = np.where(wvl[wvl<lb])[0][-1]
            end = start+length
            i+=1
    return wvl_regions, flux_regions, nmasks


def make_regions_order_revamp(wvl, flux, bounds, mask, length=400):
    '''New function just like make_regions_order, but we try a method more robust.
    In this version, I iteratively merge the regions that are the closest to one another 
    untill I reach regions of a maximum length.
    Known issue 1: the regions may end on the edges of the region. Should now be solved
    Known issue 2: the lines may be added to two regions. Solved
    '''

    ## Define the mask array that will be returned
    ## Was previously at the beggining of the function, which was false:
    ## because then the full bound will be carve in all regions.
    ## May still create duplicates and edges effects.\
    ## To correct the issue I add a condition in the last loop.
    maskarray = np.zeros(flux.shape)
    for _mask in mask:
        maskarray[(wvl>_mask[0]) & (wvl<_mask[1])] = 1

    ## Verify that the bounds are within the order
    _used_bounds = []
    for _bounds in bounds:
        lb, hb = _bounds
        if (lb>wvl[0]) & (hb<wvl[-1]):
            pass
        else:
            ## If things do not fit in the order, should I be adapting the segment?
            if lb<wvl[0]:
                lb = wvl[0]
            if hb>wvl[-1]:
                hb = wvl[-1]
        _used_bounds.append([lb, hb])

    exitloop = False
    while exitloop is False:
        ## Are there more than 1 region?
        if len(_used_bounds)<=1: exitloop; break
        ## What are the bounds that are the closest to one another?
        spaces = [_used_bounds[i][1] - _used_bounds[i+1][0] for i in range(len(_used_bounds)-1)]
        ## Now enter second loop
        counter = 0
        exit_second_loop = False
        while (exit_second_loop is False) & (counter<len(spaces)):
            ## Where is the minimum space bewteen the regions?
            ii = np.where(np.square(spaces)==np.min(np.square(spaces)))[0][0]
            ## What would be the region if we merge the closest?
            new_bound = [_used_bounds[ii][0], _used_bounds[ii+1][1]]
            ## Is this new bound less than the length?
            cond = (wvl>new_bound[0]) & (wvl<new_bound[1])
            len_new_bound = len(wvl[cond])
            ## If the new region is less than the length we want to merge them
            ## Actually, we want to have some wiggle room! So we want say, 
            ## 20% of the window. So we check that the regions are within a 80% length window
            if len_new_bound<.8*length:
                _used_bounds.remove(_used_bounds[ii])
                _used_bounds[ii] = new_bound
                ## if that is the case, we want to keep going with the main loop:
                exit_second_loop = True
            else:
                ## We want to get the second most distant spaces:
                spaces[ii] = np.inf
                ## Increase the counter to avoid running infinitely
                ## If we tried all the spaces, there is no regions to be merged anymore
                counter+=1
        if (counter>=len(spaces)-1):
            exitloop = True

    ## Now _used_bounds should contain what we want to put at the center of 400 bins windows
    ## We need to take care of the edges effect !
    wvl_regions = []; flux_regions = []; mask_regions = [];
    for i in range(len(_used_bounds)):
        cond = np.where((wvl>_used_bounds[i][0]) & (wvl<_used_bounds[i][1]))[0]
        nbbins = len(wvl[cond])
        if nbbins>length:
            # print(_used_bounds[i])
            # print('Fatal error, we try to create a region that is too large.')
            # from IPython import embed
            # embed()
            raise Exception('Fatal error, we try to create a region that is too large.')
        diff = length - nbbins
        halfdiff = diff//2
        inival = cond[0]-halfdiff 
        ## inival should be the needed offset. But there are boundary conditions!
        ## inival must not be negative !
        if inival<0:
            inival = 0
        ## inival + length should not be beyond the wavelength length !
        if inival+length>=len(wvl):
            corr = (inival+length)-len(wvl)
            inival = inival - corr
        ## Note that this will fail if the region is longer than the order !
        cond = np.arange(inival, inival+length)
        _wvl = np.zeros(length)
        _flx = np.zeros(length)
        _msk = np.zeros(length)
        _wvl = wvl[cond]
        _flx = flux[cond]
        _msk = maskarray[cond]
        ## Now we want to find the mask that is contained WITHIN THE USED BOUNDS
        cond2 = np.where((wvl[cond]<_used_bounds[i][0]) | (wvl[cond]>_used_bounds[i][1]))
        _msk[cond2] = 0

        wvl_regions.append(_wvl)
        flux_regions.append(_flx)
        mask_regions.append(_msk)
    # from IPython import embed
    # embed()
    # exit()
    return wvl_regions, flux_regions, mask_regions

def make_regions_order(wvl, flux, bounds, mask, length=400):
    '''
    Make regions from all the information on the lines and regions
    '''
    # edges = int(1/8*length)
    edge = int(1/8*length)
    edge = 0

    ## Is is possible that that one region is longer than required length
    ## We should implement something here at some point.

    ## Define the mask array that will be returned
    maskarray = np.zeros(flux.shape)
    for _mask in mask:
        maskarray[(wvl>_mask[0]) & (wvl<_mask[1])] = 1

    wvl_regions = []; flux_regions = []; mask_regions = [];
    i=0
    while i < len(bounds):
        ## Take the first bound
        lb, hb = bounds[i]
        ## Verify that the bounds are within the order
        if (lb<wvl[0]) | (hb>wvl[-1]):
            i += 1
            continue
        ## We select windows of length bins around the 
        start = np.where(wvl[wvl<lb])[0][-1] # Start as close as possible to bound
        end = start + length
        ## Have we reached the end of the order?
        if end >= len(wvl):
            end = len(wvl) - 1 - edge
            start = end - length
        hbn = hb
        ## Is there a next line? And another one after that?
        ## If yes, does it fall in the same window?
        while (hbn<wvl[end]-edge) & (i+1<(len(bounds))) \
            & (lb>=wvl[0]+edge) & (hbn<wvl[-1]-edge): ## While true, we jump bounds
            if hbn > wvl[end]: ## We don't want to go beyond the region
                break
            else:
                i += 1 ## Next iteration
                lbn, hbn = bounds[i]
            ## at this point, we may have gone too far on the right edge.
        # if hbn > wvl[end]: ## We don't want to go beyond the region
        #     print(hbn, wvl[end])
        #     i -= 1  
        #     lbn, hbn = bounds[i]
        ## Recenter the region
        nbBinsUp = np.sum(wvl[start:end] > hbn)
        nbBinsDown = np.sum(wvl[start:end] < lb)
        nbBinsDiff = nbBinsUp - nbBinsDown
        halfNbBinsDiff = nbBinsDiff // 2
        start -= halfNbBinsDiff
        end -= halfNbBinsDiff
        if start < edge:
            start = edge
            end = edge + length
        ## Append resulting regions
        wvl_regions.append(wvl[start:end])
        flux_regions.append(flux[start:end])
        _maskarray = np.copy(maskarray[start:end])
        _maskarray2 = np.zeros(maskarray[start:end].shape)
        for _mask in mask:
            if (_mask[0]>=lb) & (_mask[1]<=hbn):
                _maskarray2[(wvl[start:end]>=_mask[0]) & (wvl[start:end]<=_mask[1])] = 1
        _maskarray = _maskarray * _maskarray2
        mask_regions.append(_maskarray)
        maskarray[(wvl>=hbn) & (wvl<=hbn)] = 0
        i += 1

    return wvl_regions, flux_regions, mask_regions


def make_regions_2d_orders(wvl, flux, bounds, mask, orders, length=400):
    '''
    Wrapper used to run the make_regions function on a 2D spectrum. The function
    returns regions of given length, containing the lines requested by the 
    bounds. The function attempts to minimize the number windows without 
    changing length, and tries to avoid duplicates. 
    Input parameters:
    - wvl       :   [2D array] Wavelenth solution for the spectrum
    - flux      :   [2D array] Spectrum
    - bounds    :   list or array containing list-pair of bounds used to generate
                    a mask.
    - mask      :   Redundant with bounds. To be deleted in future versions
    - orders    :   List or array indicating in which order to search for the
                    bounds. Must be of same length as bounds.
    - length    :   Length of the output windows.
    '''
    bounds, mask, orders = np.array(bounds), np.array(mask), np.array(orders)

    ## Check that the wvl are increasing.
    firstwaves = wvl.T[0]
    if np.any(np.diff(firstwaves)<0):
        raise Exception('make_regions_2d_orders: The orders are not increasing in wavelength.')

    wvl_regions, flux_regions, nmasks = [], [], []
    for order in range(len(flux)):
        idx = orders==order
        if np.sum(idx)==0:
            continue
        _bounds = increase_width(bounds[idx], 0.05)
        _bounds = merge(_bounds) ## May be the source of problems.
        _mask = bounds[idx] ## The bounds with no increase in width, but only for this order.

        _wvl_regions, _flux_regions, _nmasks = make_regions_order_revamp(wvl[order], 
                                                            flux[order], 
                                                            _bounds, 
                                                            _mask, 
                                                            length=length)

        wvl_regions.append(_wvl_regions)
        flux_regions.append(_flux_regions)
        nmasks.append(_nmasks)

    ## Rebuilt a 1D list of regions
    nwvl_regions = []
    nflux_regions = []
    nnmasks = []
    for i in range(len(wvl_regions)):
        for j in range(len(wvl_regions[i])):
            if np.all(nmasks[i][j]==0):
                continue
            else:
                if len(wvl_regions[i][j]) < length:
                    print(len(wvl_regions[i][j]))
                else:
                    nwvl_regions.append(wvl_regions[i][j])
                    nflux_regions.append(flux_regions[i][j])
                    nnmasks.append(nmasks[i][j])

    return np.array(nwvl_regions, dtype=float), np.array(nflux_regions, dtype=float), np.array(nnmasks, dtype=float) 

def make_regions_2d(wvl, flux, bounds, mask, orders, length=400):
    '''
    Wrapper used to run the make_regions function on a 2D spectrum. The function
    returns regions of given length, containing the lines requested by the 
    bounds. The function attempts to minimize the number windows without 
    changing length, and tries to avoid duplicates. 
    Input parameters:
    - wvl       :   [2D array] Wavelenth solution for the spectrum
    - flux      :   [2D array] Spectrum
    - bounds    :   list or array containing list-pair of bounds used to generate
                    a mask.
    - mask      :   Redundant with bounds. To be deleted in future versions
    - orders    :   List or array indicating in which order to search for the
                    bounds. Must be of same length as bounds.
    - length    :   Length of the output windows.
    '''
    bounds, mask, orders = np.array(bounds), np.array(mask), np.array(orders)

    wvl_regions, flux_regions, nmasks = [], [], []

    known_orders = []
    for order in orders:
        if order in known_orders:
            continue
        known_orders.append(order)
        vals = np.where(orders==order)
        _bounds = increase_width(bounds[vals])
        _bounds = merge(_bounds)
        _mask = merge(mask)
        _wvl_regions, _flux_regions, _nmasks = make_regions(wvl[order], 
                                                            flux[order], 
                                                            _bounds, 
                                                            _mask, 
                                                            length=length)
        wvl_regions.append(_wvl_regions)
        flux_regions.append(_flux_regions)
        nmasks.append(_nmasks)

    nwvl_regions = []
    nflux_regions = []
    nnmasks = []
    for i in range(len(wvl_regions)):
        for j in range(len(wvl_regions[i])):
            if np.all(nmasks[i][j]==0):
                continue
            else:
                nwvl_regions.append(wvl_regions[i][j])
                nflux_regions.append(flux_regions[i][j])
                nnmasks.append(nmasks[i][j])

    return np.array(nwvl_regions), np.array(nflux_regions), np.array(nnmasks) 

def read_lines(filename=None, returnall=False):
    if filename is None:
        # filename = paths.irap_tools_data_path+'selected_line_list.txt'
        raise Exception('read_lines: No input file')
    f = open(filename, 'r')
    w, lb, hb, label, ion, orders = [], [], [], [], [], []
    for line in f.readlines():
        if line.strip()[0]=="#": continue
        if line.strip()=="": continue ## Empty line
        # print(line)
        l = line.split()
        w.append(float(l[0]))
        lb.append(float(l[1]))
        hb.append(float(l[2]))
        label.append(l[3])
        ion.append(int(l[4]))
        orders.append(int(l[5]))
    f.close()
    # # Enlarge regions to take some continuum points
    bounds = []
    for i in range(len(hb)):
        _lb = lb[i]
        _hb = hb[i]
        bounds.append([_lb, _hb])

    ## Create the mask actually use for the analysis
    mask = [[lb[i], hb[i]] for i in range(len(lb))]
    if returnall:
        listobj = [np.array(w), np.array(lb), np.array(hb), label, np.array(ion), \
                np.array(orders)]
        outdict = {'wvls':listobj[0], 'lbs':listobj[1], 'hbs':listobj[2],
                   'labels':listobj[3], 'ions':listobj[4], 'orders':listobj[5]}
        return outdict
    else:
        return np.array(bounds), np.array(orders)


def make_contrained_regions(wvl, flux, regions):
    nwvl = []
    nflux = []
    ## let's take additional 10 km/s on each side
    shift = 5/(3*1e5)
    for region in regions:
        idx = ((wvl>(region[0]-region[0]*shift)) 
               & (wvl<(region[1]+region[1]*shift)))
        if len(wvl[idx]) != len(flux[idx]):
            raise Exception('Problem generating regions') 
        nwvl.append(wvl[idx])
        nflux.append(flux[idx])
    return nwvl, nflux

def read_mask(file):
    '''returns the wavelength and associated line name'''
    f = open(file, 'r')
    wvls = []; labels = []; ions = [];
    for line in f.readlines():
        wvls.append(float(line.split()[0]))
        labels.append(line.split()[3])
        ions.append(int(float(line.split()[4])))
    f.close()
    return wvls, labels, ions