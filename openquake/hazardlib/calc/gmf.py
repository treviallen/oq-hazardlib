# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2016 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module :mod:`~openquake.hazardlib.calc.gmf` exports
:func:`ground_motion_fields`.
"""
import collections

import numpy
import scipy.stats

from openquake.baselib.general import groupby2
from openquake.baselib.python3compat import zip
from openquake.hazardlib.const import StdDev
from openquake.hazardlib.calc import filters
from openquake.hazardlib.gsim.base import ContextMaker
from openquake.hazardlib.imt import from_string
from openquake.hazardlib import site

U8 = numpy.uint8
U16 = numpy.uint16
U32 = numpy.uint32
F32 = numpy.float32

U8SIZE = 2 ** 8
U16SIZE = 2 ** 16
U32SIZE = 2 ** 32


def gmv_dt(imts):
    """
    Build the numpy dtype for the ground motion values from the IMTs;
    it has fields 'sid', 'eid' and 'gmv' where 'gmv' is a composite type
    depending on the intensity measure types.

    :param imts: an ordered list of IMT strings
    """
    imt_dt = numpy.dtype([(imt, F32) for imt in imts])
    return numpy.dtype([('sid', U32), ('eid', U32), ('gmv', imt_dt)])


class CorrelationButNoInterIntraStdDevs(Exception):
    def __init__(self, corr, gsim):
        self.corr = corr
        self.gsim = gsim

    def __str__(self):
        return '''\
You cannot use the correlation model %s with the GSIM %s, \
that defines only the total standard deviation. If you want to use a \
correlation model you have to select a GMPE that provides the inter and \
intra event standard deviations.''' % (
            self.corr.__class__.__name__, self.gsim.__class__.__name__)


class GmfComputer(object):
    """
    Given an earthquake rupture, the ground motion field computer computes
    ground shaking over a set of sites, by randomly sampling a ground
    shaking intensity model.

    :param :class:`openquake.hazardlib.source.rupture.Rupture` rupture:
        Rupture to calculate ground motion fields radiated from.

    :param :class:`openquake.hazardlib.site.SiteCollection` sites:
        Sites of interest to calculate GMFs.

    :param gmv_dt:
        a nested numpy dtype with the form (sid, eid, gmv: (imt1, ...))

    :param truncation_level:
        Float, number of standard deviations for truncation of the intensity
        distribution, or ``None``.

    :param correlation_model:
        Instance of correlation model object. See
        :mod:`openquake.hazardlib.correlation`. Can be ``None``, in which
        case non-correlated ground motion fields are calculated.
        Correlation model is not used if ``truncation_level`` is zero.
    """
    def __init__(self, rupture, sites, gmv_dt, gsims,
                 truncation_level=None, correlation_model=None):
        assert sites, sites
        self.rupture = rupture
        self.sites = sites
        self.imts = [from_string(imt) for imt in gmv_dt['gmv'].names]
        self.gsims = gsims
        self.truncation_level = truncation_level
        self.correlation_model = correlation_model
        self.ctx = ContextMaker(gsims).make_contexts(sites, rupture)
        self.gmv_dt = gmv_dt  # numpy dtype used in event based calculations

    def _compute(self, seed, gsim, realizations):
        # the method doing the real stuff; use compute instead
        if seed is not None:
            numpy.random.seed(seed)
        result = collections.OrderedDict()
        sctx, rctx, dctx = self.ctx

        if self.truncation_level == 0:
            assert self.correlation_model is None
            for imt in self.imts:
                mean, _stddevs = gsim.get_mean_and_stddevs(
                    sctx, rctx, dctx, imt, stddev_types=[])
                mean = gsim.to_imt_unit_values(mean)
                mean.shape += (1, )
                mean = mean.repeat(realizations, axis=1)
                result[str(imt)] = mean
            return result
        elif self.truncation_level is None:
            distribution = scipy.stats.norm()
        else:
            assert self.truncation_level > 0
            distribution = scipy.stats.truncnorm(
                - self.truncation_level, self.truncation_level)

        for imt in self.imts:
            if gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES == \
               set([StdDev.TOTAL]):
                # If the GSIM provides only total standard deviation, we need
                # to compute mean and total standard deviation at the sites
                # of interest.
                # In this case, we also assume no correlation model is used.
                if self.correlation_model:
                    raise CorrelationButNoInterIntraStdDevs(
                        self.correlation_model, gsim)

                mean, [stddev_total] = gsim.get_mean_and_stddevs(
                    sctx, rctx, dctx, imt, [StdDev.TOTAL])
                stddev_total = stddev_total.reshape(stddev_total.shape + (1, ))
                mean = mean.reshape(mean.shape + (1, ))

                total_residual = stddev_total * distribution.rvs(
                    size=(len(self.sites), realizations))
                gmf = gsim.to_imt_unit_values(mean + total_residual)
            else:
                mean, [stddev_inter, stddev_intra] = gsim.get_mean_and_stddevs(
                    sctx, rctx, dctx, imt,
                    [StdDev.INTER_EVENT, StdDev.INTRA_EVENT])
                stddev_intra = stddev_intra.reshape(stddev_intra.shape + (1, ))
                stddev_inter = stddev_inter.reshape(stddev_inter.shape + (1, ))
                mean = mean.reshape(mean.shape + (1, ))

                intra_residual = stddev_intra * distribution.rvs(
                    size=(len(self.sites), realizations))

                if self.correlation_model is not None:
                    ir = self.correlation_model.apply_correlation(
                        self.sites, imt, intra_residual)
                    # this fixes a mysterious bug: ir[row] is actually
                    # a matrix of shape (E, 1) and not a vector of size E
                    intra_residual = numpy.zeros(ir.shape)
                    for i, val in numpy.ndenumerate(ir):
                        intra_residual[i] = val

                inter_residual = stddev_inter * distribution.rvs(
                    size=realizations)

                gmf = gsim.to_imt_unit_values(
                    mean + intra_residual + inter_residual)

            result[str(imt)] = gmf

        return result

    def compute(self, seed, gsim, eids):
        """
        Compute a ground motion array for the given sites.

        :param seed:
            seed for the numpy random number generator
        :param gsim:
            a GSIM instance
        :param:
            event IDs, a list of integers
        :returns:
            a numpy array of dtype gmv_dt and size num_events * num_sites
        """
        multiplicity = len(eids)
        sids = self.sites.sids
        gmfa = numpy.zeros(len(self.sites) * multiplicity, self.gmv_dt)
        for imt, gmfarray in self._compute(seed, gsim, multiplicity).items():
            i = 0
            for sid, gmvs in zip(sids, gmfarray):
                for eid, gmv in zip(eids, F32(gmvs)):
                    rec = gmfa[i]
                    rec['sid'] = sid
                    rec['eid'] = eid
                    rec['gmv'][imt] = gmv
                    i += 1
        return gmfa

    # this is much faster than .compute
    def calcgmfs(self, multiplicity, seed, rlzs_by_gsim=None):
        """
        Compute the ground motion fields for the given gsims, sites,
        multiplicity and seed.

        :param multiplicity:
            the number of GMFs to return
        :param seed:
            seed for the numpy random number generator
        :param rlzs_by_gsim:
            a dictionary {gsim instance: realization indices}
        :returns:
            a dictionary rlz -> imt -> array(N, M)
        """
        ddic = {}  # rlz -> imt -> array(N, M)
        for i, gsim in enumerate(self.gsims):
            rlz_ids = [i] if rlzs_by_gsim is None else rlzs_by_gsim[gsim]
            for r, rlz in enumerate(rlz_ids):
                ddic[rlz] = self._compute(seed + r, gsim, multiplicity)
        return ddic

# #################################################################### #


class GmfExtractor(object):
    """
    A class used to filter the GMF array. Here is an example:

    >>> extractor = GmfExtractor(range(5), ['PGA', 'SA(0.1)'])
    >>> gmfa = numpy.array([
    ...     (0, 1, 0, 42, 0.030),
    ...     (0, 1, 0, 43, 0.031),
    ...     (0, 2, 0, 43, 0.032),
    ...     (0, 2, 0, 42, 0.033),
    ...     (2, 1, 0, 42, 0.034),
    ...     (2, 1, 1, 43, 0.035),
    ...     (2, 1, 1, 44, 0.036),
    ...     ], extractor.gmv_dt)
    >>> print(extractor(gmfa, sid=0, rlzi=1, imti=0)['gmv'])
    [ 0.03   0.031]
    """
    gmv_dt = numpy.dtype([('sid', U16), ('rlzi', U16), ('imti', U8),
                          ('eid', U32), ('gmv', F32)])
    KNOWN_NAMES = set(['sid', 'rlzi', 'imti', 'eid'])

    def __init__(self, sitecol, imts, rlzs_assoc=None):
        if len(sitecol) > U16SIZE:
            raise ValueError(
                'the number of sites is %d, the upper limit is %d' %
                (len(sitecol), U16SIZE))
        if len(imts) > U8SIZE:
            raise ValueError(
                'the number of IMTs is %d, the upper limit is %d' %
                (len(imts), U8SIZE))
        if rlzs_assoc and len(rlzs_assoc.realizations) > U16SIZE:
            raise ValueError(
                'the number of realizations is %d, the upper limit is %d' %
                (len(rlzs_assoc.realizations), U16SIZE))
        self.sitecol = sitecol
        self.imts = imts
        self.imt2idx = {imt: i for i, imt in enumerate(imts)}
        self.rlzs_assoc = rlzs_assoc

    def imtdict2array(self, iml):
        """
        Returns an array of float32 from a dictionary imt -> value
        """
        array = numpy.zeros(len(self.imts), F32)
        if iml:
            for imt, value in iml.items():
                array[self.imt2idx[imt]] = value
        return array

    def __call__(self, array, **kw):
        """
        Extract a subarray by filtering on given keyword arguments
        """
        for name, value in kw.items():
            if name not in self.KNOWN_NAMES:
                raise ValueError('%s is not one of %s' %
                                 (name, self.KNOWN_NAMES))
            array = array[array[name] == value]
        return array

    def by_rlz(self, array):
        """
        Group a composite array with fields 'rlzi', 'eid', 'gmv' by
        'rlzi' and returns a dictionary rlz -> [(eid, gmv), ...]
        """
        rlzs = self.rlzs_assoc.realizations
        return {rlzs[i]: data for i, data in groupby2(
            array, 'rlzi', ('eid', 'gmv'))}


class Computer(object):
    """
    Given a rich rupture, computes the ground shaking over a set of sites,
    by randomly sampling a ground shaking intensity model.

    :param :class:`openquake.hazardlib.source.rupture.Rupture` rupture:
        Rupture to calculate ground motion fields radiated from.

    :param :class:`openquake.hazardlib.site.SiteCollection` sites:
        Sites of interest to calculate GMFs.

    :param truncation_level:
        Float, number of standard deviations for truncation of the intensity
        distribution, or ``None``.

    :param correlation_model:
        Instance of correlation model object. See
        :mod:`openquake.hazardlib.correlation`. Can be ``None``, in which
        case non-correlated ground motion fields are calculated.
        Correlation model is not used if ``truncation_level`` is zero.
    """
    def __init__(self, rupture, extractor,
                 truncation_level=None, correlation_model=None):
        self.extractor = extractor
        self.rupture = rupture
        self.sites = site.FilteredSiteCollection(
            rupture.indices, extractor.sitecol)
        gsims = extractor.rlzs_assoc.gsims_by_trt_id[rupture.trt_id]
        self.truncation_level = truncation_level
        self.correlation_model = correlation_model
        self.ctx = ContextMaker(gsims).make_contexts(
            self.sites, rupture.rupture)
        self.imts = [from_string(imt) for imt in extractor.imts]

    def _compute(self, seed, gsim, realizations):
        # the method doing the real stuff; use compute instead
        if seed is not None:
            numpy.random.seed(seed)
        imts = self.imts
        result = numpy.zeros((len(imts), len(self.sites), realizations))
        imt2idx = self.extractor.imt2idx
        sctx, rctx, dctx = self.ctx

        if self.truncation_level == 0:
            assert self.correlation_model is None
            for imt in imts:
                mean, _stddevs = gsim.get_mean_and_stddevs(
                    sctx, rctx, dctx, imt, stddev_types=[])
                mean = gsim.to_imt_unit_values(mean)
                mean.shape += (1, )
                mean = mean.repeat(realizations, axis=1)
                result[imt2idx[str(imt)]] = mean
            return result
        elif self.truncation_level is None:
            distribution = scipy.stats.norm()
        else:
            assert self.truncation_level > 0
            distribution = scipy.stats.truncnorm(
                - self.truncation_level, self.truncation_level)

        for imt in imts:
            if gsim.DEFINED_FOR_STANDARD_DEVIATION_TYPES == \
               set([StdDev.TOTAL]):
                # If the GSIM provides only total standard deviation, we need
                # to compute mean and total standard deviation at the sites
                # of interest.
                # In this case, we also assume no correlation model is used.
                if self.correlation_model:
                    raise CorrelationButNoInterIntraStdDevs(
                        self.correlation_model, gsim)

                mean, [stddev_total] = gsim.get_mean_and_stddevs(
                    sctx, rctx, dctx, imt, [StdDev.TOTAL])
                stddev_total = stddev_total.reshape(stddev_total.shape + (1, ))
                mean = mean.reshape(mean.shape + (1, ))

                total_residual = stddev_total * distribution.rvs(
                    size=(len(self.sites), realizations))
                gmf = gsim.to_imt_unit_values(mean + total_residual)
            else:
                mean, [stddev_inter, stddev_intra] = gsim.get_mean_and_stddevs(
                    sctx, rctx, dctx, imt,
                    [StdDev.INTER_EVENT, StdDev.INTRA_EVENT])
                stddev_intra = stddev_intra.reshape(stddev_intra.shape + (1, ))
                stddev_inter = stddev_inter.reshape(stddev_inter.shape + (1, ))
                mean = mean.reshape(mean.shape + (1, ))

                intra_residual = stddev_intra * distribution.rvs(
                    size=(len(self.sites), realizations))

                if self.correlation_model is not None:
                    ir = self.correlation_model.apply_correlation(
                        self.sites, imt, intra_residual)
                    # this fixes a mysterious bug: ir[row] is actually
                    # a matrix of shape (E, 1) and not a vector of size E
                    intra_residual = numpy.zeros(ir.shape)
                    for i, val in numpy.ndenumerate(ir):
                        intra_residual[i] = val

                inter_residual = stddev_inter * distribution.rvs(
                    size=realizations)

                gmf = gsim.to_imt_unit_values(
                    mean + intra_residual + inter_residual)

            result[imt2idx[str(imt)]] = gmf

        return result

    def compute(self, min_iml):
        """
        Compute the ground motion fields generated by the underlying rupture.
        :returns:
            a numpy array of type gmv_dt
        """
        rupture = self.rupture
        records = []
        min_iml = self.extractor.imtdict2array(min_iml)
        for (trt_id, gsim), rlzs in self.extractor.rlzs_assoc.items():
            for r, rlz in enumerate(rlzs):
                gmf_by_imt = self._compute(
                    rupture.rupture.seed + r, gsim, rupture.multiplicity)
                for imti, array in enumerate(gmf_by_imt):
                    for sid, gmvs in zip(rupture.indices, array):
                        for gmv, eid in zip(gmvs, rupture.eids):
                            if gmv >= min_iml[imti]:
                                records.append(
                                    (sid, rlz.ordinal, imti, eid, gmv))
        return numpy.array(records, self.extractor.gmv_dt)


# this is not used in the engine; it is still useful for usage in IPython
# when demonstrating hazardlib capabilities
def ground_motion_fields(rupture, sites, imts, gsim, truncation_level,
                         realizations, correlation_model=None,
                         rupture_site_filter=filters.rupture_site_noop_filter,
                         seed=None):
    """
    Given an earthquake rupture, the ground motion field calculator computes
    ground shaking over a set of sites, by randomly sampling a ground shaking
    intensity model. A ground motion field represents a possible 'realization'
    of the ground shaking due to an earthquake rupture. If a non-trivial
    filtering function is passed, the final result is expanded and filled
    with zeros in the places corresponding to the filtered out sites.

    .. note::

     This calculator is using random numbers. In order to reproduce the
     same results numpy random numbers generator needs to be seeded, see
     http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.seed.html

    :param openquake.hazardlib.source.rupture.Rupture rupture:
        Rupture to calculate ground motion fields radiated from.
    :param openquake.hazardlib.site.SiteCollection sites:
        Sites of interest to calculate GMFs.
    :param imts:
        List of intensity measure type objects (see
        :mod:`openquake.hazardlib.imt`).
    :param gsim:
        Ground-shaking intensity model, instance of subclass of either
        :class:`~openquake.hazardlib.gsim.base.GMPE` or
        :class:`~openquake.hazardlib.gsim.base.IPE`.
    :param truncation_level:
        Float, number of standard deviations for truncation of the intensity
        distribution, or ``None``.
    :param realizations:
        Integer number of GMF realizations to compute.
    :param correlation_model:
        Instance of correlation model object. See
        :mod:`openquake.hazardlib.correlation`. Can be ``None``, in which case
        non-correlated ground motion fields are calculated. Correlation model
        is not used if ``truncation_level`` is zero.
    :param rupture_site_filter:
        Optional rupture-site filter function. See
        :mod:`openquake.hazardlib.calc.filters`.
    :param int seed:
        The seed used in the numpy random number generator
    :returns:
        Dictionary mapping intensity measure type objects (same
        as in parameter ``imts``) to 2d numpy arrays of floats,
        representing different realizations of ground shaking intensity
        for all sites in the collection. First dimension represents
        sites and second one is for realizations.
    """
    ruptures_sites = list(rupture_site_filter([(rupture, sites)]))
    if not ruptures_sites:
        return dict((imt, numpy.zeros((len(sites), realizations)))
                    for imt in imts)
    [(rupture, sites)] = ruptures_sites
    imt_dt = numpy.dtype([(str(imt), F32) for imt in imts])
    gmv_dt = numpy.dtype([('sid', U32), ('eid', U32), ('gmv', imt_dt)])
    gc = GmfComputer(rupture, sites, gmv_dt, [gsim],
                     truncation_level, correlation_model)
    result = gc._compute(seed, gsim, realizations)
    for imt, gmf in result.items():
        # makes sure the lenght of the arrays in output is the same as sites
        if rupture_site_filter is not filters.rupture_site_noop_filter:
            result[imt] = sites.expand(gmf, placeholder=0)

    return {from_string(imt): result[imt] for imt in result}
