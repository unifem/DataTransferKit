//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \file   DTK_MovingLeastSquareReconstructionOperator_impl.hpp
 * \author Stuart R. Slattery
 * \brief  Moving least square interpolator.
 */
//---------------------------------------------------------------------------//

#ifndef DTK_MOVINGLEASTSQUARERECONSTRUCTIONOPERATOR_IMPL_HPP
#define DTK_MOVINGLEASTSQUARERECONSTRUCTIONOPERATOR_IMPL_HPP

#include <algorithm>
#include <iostream>
#include <limits>

#include "DTK_BasicEntityPredicates.hpp"
#include "DTK_CenterDistributor.hpp"
#include "DTK_DBC.hpp"
#include "DTK_LocalMLSProblem.hpp"
#include "DTK_MovingLeastSquareReconstructionOperator.hpp"
#include "DTK_PredicateComposition.hpp"
#include "DTK_SplineInterpolationPairing.hpp"

#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Ptr.hpp>

#include <Tpetra_MultiVector.hpp>

// added QC
#ifdef TUNING_INDICATOR_VALUES
#include <fstream>
#endif
// added QC

namespace DataTransferKit
{
//---------------------------------------------------------------------------//
// Constructor.
template <class Basis, int DIM>
MovingLeastSquareReconstructionOperator<Basis, DIM>::
    MovingLeastSquareReconstructionOperator(
        const Teuchos::RCP<const TpetraMap> &domain_map,
        const Teuchos::RCP<const TpetraMap> &range_map,
        const Teuchos::ParameterList &parameters )
    : Base( domain_map, range_map )
    , d_use_knn( false )
    , d_knn( 0 )
    , d_radius( 0.0 )
    , d_domain_entity_dim( 0 )
    , d_range_entity_dim( 0 )
    // added by QC
    , d_leaf( 0 )
    , d_use_qrcp( false )
    , d_sigma( 2.0 )
    , d_do_post( false )
    , d_rho( -1.0 )
#ifdef TUNING_INDICATOR_VALUES
    , d_file_name( "" )
#endif
// added by QC
{
    // Determine if we are doing kNN search or radius search.
    if ( parameters.isParameter( "Type of Search" ) )
    {
        if ( "Radius" == parameters.get<std::string>( "Type of Search" ) )
        {
            d_use_knn = false;
        }
        else if ( "Nearest Neighbor" ==
                  parameters.get<std::string>( "Type of Search" ) )
        {
            d_use_knn = true;
        }
        else
        {
            // Otherwise we got an invalid search type.
            DTK_INSIST( false );
        }
    }

    // If we are doing kNN support get the number of neighbors.
    if ( d_use_knn )
    {
        DTK_REQUIRE( parameters.isParameter( "Num Neighbors" ) );
        d_knn = parameters.get<int>( "Num Neighbors" );
    }

    // Otherwise we are doing the radius search so get the basis radius.
    else
    {
        DTK_REQUIRE( parameters.isParameter( "RBF Radius" ) );
        d_radius = parameters.get<double>( "RBF Radius" );
    }

    // Get the topological dimension of the domain and range entities. This
    // map will use their centroids for the point cloud.
    if ( parameters.isParameter( "Domain Entity Dimension" ) )
    {
        d_domain_entity_dim = parameters.get<int>( "Domain Entity Dimension" );
    }
    if ( parameters.isParameter( "Range Entity Dimension" ) )
    {
        d_range_entity_dim = parameters.get<int>( "Range Entity Dimension" );
    }
    // added, QC
    if ( parameters.isParameter( "Leaf Size" ) )
    {
        d_leaf = parameters.get<int>( "Leaf Size" );
    }
    if ( parameters.isParameter( "Use QRCP Impl" ) )
    {
        d_use_qrcp = parameters.get<bool>( "Use QRCP Impl" );
    }
    if ( parameters.isParameter( "Indicator Threshold" ) )
    {
        d_sigma = parameters.get<double>( "Indicator Threshold" );
    }
    if ( d_sigma <= 0.0 )
        d_sigma = 2.0; // this might be too small
    if ( parameters.isParameter( "Resolve Discontinuity" ) )
    {
        d_do_post = parameters.get<bool>( "Resolve Discontinuity" );
    }
    if ( parameters.isParameter( "Local Rho Scaling" ) )
    {
        d_rho = parameters.get<double>( "Local Rho Scaling" );
    }
#ifdef TUNING_INDICATOR_VALUES
    if ( parameters.isParameter( "Indicator Output File" ) )
    {
        d_file_name = parameters.get<std::string>( "Indicator Output File" );
    }
#endif
    // added, QC
}

//---------------------------------------------------------------------------//
// Setup the map operator.
template <class Basis, int DIM>
void MovingLeastSquareReconstructionOperator<Basis, DIM>::setupImpl(
    const Teuchos::RCP<FunctionSpace> &domain_space,
    const Teuchos::RCP<FunctionSpace> &range_space )
{
    DTK_REQUIRE( Teuchos::nonnull( domain_space ) );
    DTK_REQUIRE( Teuchos::nonnull( range_space ) );

    // Extract the Support maps.
    const Teuchos::RCP<const typename Base::TpetraMap> domain_map =
        this->getDomainMap();
    const Teuchos::RCP<const typename Base::TpetraMap> range_map =
        this->getRangeMap();

    // Get the parallel communicator.
    Teuchos::RCP<const Teuchos::Comm<int>> comm = domain_map->getComm();

    // Determine if we have range and domain data on this process.
    bool nonnull_domain = Teuchos::nonnull( domain_space->entitySet() );
    bool nonnull_range = Teuchos::nonnull( range_space->entitySet() );

    // remove unused warning...
    (void)nonnull_domain;
    (void)nonnull_range;

    // Extract the source nodes and their ids.
    Teuchos::ArrayRCP<double> source_centers;
    Teuchos::ArrayRCP<GO> source_support_ids;
    getNodeCoordsAndIds( domain_space, d_domain_entity_dim, source_centers,
                         source_support_ids );

    // Extract the target nodes and their ids.
    Teuchos::ArrayRCP<double> target_centers;
    Teuchos::ArrayRCP<GO> target_support_ids;
    getNodeCoordsAndIds( range_space, d_range_entity_dim, target_centers,
                         target_support_ids );

    // Calculate an approximate neighborhood distance for the local target
    // centers. If using kNN, compute an approximation. If doing a radial
    // search, use the radius. We will use these distances to expand the local
    // bounding box to ensure we find all of our neighbors in parallel.
    double target_proximity = 0.0;
    if ( d_use_knn )
    {
        // Get the local bounding box.
        Teuchos::Tuple<double, 6> local_box;
        range_space->entitySet()->localBoundingBox( local_box );

        // Calculate the largest span of the cardinal directions.
        target_proximity = local_box[3] - local_box[0];
        for ( int d = 1; d < DIM; ++d )
        {
            target_proximity =
                std::max( target_proximity, local_box[d + 3] - local_box[d] );
        }

        // Take the proximity to be 10% of the largest distance.
        target_proximity *= 0.1;
    }
    else
    {
        target_proximity = d_radius;
    }

    // Gather the source centers that are in the proximity of the target
    // centers on this proc.
    Teuchos::Array<double> dist_sources;
    // added by QC
    d_dist = Teuchos::rcp(
        new CenterDistributor<DIM>( comm, source_centers(), target_centers(),
                                    target_proximity, dist_sources ) );
    CenterDistributor<DIM> &distributor = *d_dist;
    // CenterDistributor<DIM> distributor( comm, source_centers(),
    //                                     target_centers(), target_proximity,
    //                                     dist_sources );

    // Gather the global ids of the source centers that are within the proximity
    // of
    // the target centers on this proc.
    Teuchos::Array<GO> dist_source_support_ids( distributor.getNumImports() );
    Teuchos::ArrayView<const GO> source_support_ids_view = source_support_ids();
    distributor.distribute( source_support_ids_view,
                            dist_source_support_ids() );

    // Build the source/target pairings.
    // added the leaf parameter, QC
    d_pairings = Teuchos::rcp( new SplineInterpolationPairing<DIM>(
        dist_sources, target_centers(), d_use_knn, d_knn, d_radius, d_leaf,
        d_use_qrcp ) );
    SplineInterpolationPairing<DIM> &pairings = *d_pairings;
    // SplineInterpolationPairing<DIM> pairings(
    //     dist_sources, target_centers(), d_use_knn, d_knn, d_radius, d_leaf );

    // Build the basis.
    Teuchos::RCP<Basis> basis = BP::create();

    // Build the interpolation matrix.
    Teuchos::ArrayRCP<SupportId> children_per_parent =
        pairings.childrenPerParent();
    // SupportId is unsigned long
    SupportId max_entries_per_row = *std::max_element(
        children_per_parent.begin(), children_per_parent.end() );
    d_coupling_matrix = Teuchos::rcp( new Tpetra::CrsMatrix<Scalar, LO, GO>(
        range_map, max_entries_per_row ) );
    Teuchos::ArrayView<const double> target_view;
    Teuchos::Array<GO> indices( max_entries_per_row );
    Teuchos::ArrayView<const double> values;
    Teuchos::ArrayView<const unsigned> pair_gids;
    int nn = 0;
    int local_num_tgt = target_support_ids.size();
    for ( int i = 0; i < local_num_tgt; ++i )
    {
        // If there is no support for this target center then do not build a
        // local basis.
        if ( 0 < pairings.childCenterIds( i ).size() )
        {
            // Get a view of this target center.
            target_view = target_centers( i * DIM, DIM );

            // Build the local interpolation problem.
            LocalMLSProblem<Basis, DIM> local_problem(
                target_view, pairings.childCenterIds( i ), dist_sources, *basis,
                pairings.parentSupportRadius( i ), d_use_qrcp, d_rho );

            // added QC, set the real radius
            pairings.setRadius( i, local_problem.r() );

            // Get MLS shape function values for this target point.
            values = local_problem.shapeFunction();
            nn = values.size();

            // added QC, set the real nn
            pairings.setSize( i, nn );

            // Populate the interpolation matrix row.
            pair_gids = pairings.childCenterIds( i );
            for ( int j = 0; j < nn; ++j )
            {
                indices[j] = dist_source_support_ids[pair_gids[j]];
            }
            d_coupling_matrix->insertGlobalValues( target_support_ids[i],
                                                   indices( 0, nn ), values );
        }
    }
    d_coupling_matrix->fillComplete( domain_map, range_map );
    DTK_ENSURE( d_coupling_matrix->isFillComplete() );
}

//---------------------------------------------------------------------------//
// Apply the operator.
template <class Basis, int DIM>
void MovingLeastSquareReconstructionOperator<Basis, DIM>::applyImpl(
    const TpetraMultiVector &X, TpetraMultiVector &Y, Teuchos::ETransp mode,
    double alpha, double beta ) const
{
    d_coupling_matrix->apply( X, Y, mode, alpha, beta );

    // added QC
    // post-processing
    if ( d_use_qrcp && d_do_post )
    {
        Teuchos::SerialDenseMatrix<LO, double> domainDistV;
        // send source to distributed target map
        sendSource2TargetMap( X, domainDistV );
        // fixing
        detectResolveDisc( domainDistV, Y );
    }
    // added QC
}

//---------------------------------------------------------------------------//
// Transpose apply option.
template <class Basis, int DIM>
bool MovingLeastSquareReconstructionOperator<Basis,
                                             DIM>::hasTransposeApplyImpl() const
{
    return true;
}

//---------------------------------------------------------------------------//
// Extract node coordinates and ids from an iterator.
template <class Basis, int DIM>
void MovingLeastSquareReconstructionOperator<Basis, DIM>::getNodeCoordsAndIds(
    const Teuchos::RCP<FunctionSpace> &space, const int entity_dim,
    Teuchos::ArrayRCP<double> &centers,
    Teuchos::ArrayRCP<GO> &support_ids ) const
{
    // Get an iterator over the local nodes.
    EntityIterator iterator;
    if ( Teuchos::nonnull( space->entitySet() ) )
    {
        LocalEntityPredicate local_predicate(
            space->entitySet()->communicator()->getRank() );
        PredicateFunction predicate = PredicateComposition::And(
            space->selectFunction(), local_predicate.getFunction() );
        iterator = space->entitySet()->entityIterator( entity_dim, predicate );
    }

    // Extract the coordinates and support ids of the nodes.
    int local_num_node = iterator.size();
    centers = Teuchos::ArrayRCP<double>( DIM * local_num_node );
    support_ids = Teuchos::ArrayRCP<GO>( local_num_node );
    Teuchos::Array<SupportId> node_supports;
    EntityIterator begin = iterator.begin();
    EntityIterator end = iterator.end();
    int entity_counter = 0;
    for ( EntityIterator entity = begin; entity != end;
          ++entity, ++entity_counter )
    {
        space->shapeFunction()->entitySupportIds( *entity, node_supports );
        DTK_CHECK( 1 == node_supports.size() );
        support_ids[entity_counter] = node_supports[0];
        space->localMap()->centroid( *entity,
                                     centers( DIM * entity_counter, DIM ) );
    }
}

// added by QC
// impl of detecting & resolving disc

// step I send source values to target decomp map

template <class Basis, int DIM>
void MovingLeastSquareReconstructionOperator<Basis, DIM>::sendSource2TargetMap(
    const TpetraMultiVector &domainV,
    Teuchos::SerialDenseMatrix<LO, double> &domainDistV ) const
{
    // get the number of dimensions in the solution
    const int col = domainV.getNumVectors();

    // get the number of rows in the distributor
    const int row = d_dist->getNumImports();

    // reshape the dense matrix
    domainDistV.reshape( row, col );

    // handy
    typedef Teuchos::ArrayRCP<const double> cview_t;
    typedef Teuchos::ArrayView<double> view_t;

    // NOTE here we assume the local index aligns with the global ordering
    // in multivector, which should be fine??

    for ( int dim = 0; dim < col; ++dim )
    {
        // create const view on current dimension
        cview_t source_view = domainV.getData( dim );
        // create non-const view on current distributed domain
        view_t dist_source_view = view_t( domainDistV[dim], row );

        // send here
        d_dist->distribute( source_view(), dist_source_view );
    }
}

template <class Basis, int DIM>
typename MovingLeastSquareReconstructionOperator<Basis, DIM>::LO
MovingLeastSquareReconstructionOperator<Basis, DIM>::detectResolveDisc(
    const Teuchos::SerialDenseMatrix<LO, double> &domainDistV,
    TpetraMultiVector &rangeIntrmV ) const
{
    // handy
    typedef Teuchos::ArrayView<const double> cview_t;
    typedef Teuchos::ArrayView<double> view_t;
    typedef Teuchos::ArrayView<const unsigned> stencil_t;

    // machine precision
    const static double eps = std::numeric_limits<double>::epsilon();
    const double sigma = d_sigma;

    // implement the smoothness indicator in a lambda
    // input parameters:
    //      srcs: complete source value view
    //      stncl: local stencil, which might be larger
    //      nn: actual stencil size
    //      tar: target intermediate solution
    //      h: closest distance from target center to source stencil
    //      hh: the actual stencil radii
    const auto compute_indicator_value =
        [=]( const cview_t &srcs, const stencil_t &stncl, const int nn,
             const double tar, const double h, const double hh ) -> double {
        //=============================================================
        // step 1, compute the largest abs value in sources and target
        //=============================================================

        double src_max = 0.0;
        for ( int i = 0; i < nn; ++i )
        {
            src_max = std::max( src_max, std::abs( srcs[stncl[i]] ) );
        }
        // treatment 1, if the max value is exactly zero, return true
        // regardless the value of h
        const double max_v = std::max( src_max, std::abs( tar ) );
        if ( max_v == 0.0 )
            return 0.0;

        // if src_max is zero
        if ( src_max == 0.0 )
            src_max = max_v;

        //===============================================
        // step 2, get the closest source function value
        //===============================================

        const double src = srcs[stncl[0]];
        // normalize the function values
        const double src_nrm = src / src_max;
        const double tar_nrm = tar / src_max;

        // treatment 2, if src_nrm and tar_nrm are too close to each other
        // return true regardless h, avoiding cancellation
        const double diff_abs = std::abs( src_nrm - tar_nrm );
        if ( diff_abs <= 10.0 * eps )
            return 0.0;

        //===============================================
        // step 3, Compute differentiation
        //===============================================

        // treatment 3, if h is too close to zero, direct analyze the
        // function difference
        if ( h / hh <= 10.0 * eps )
        {
            // in this case, it means that the transfer is happened to be
            // fitting to the source center, which should be high order
            // if the function is smooth, otherwise, the function value
            // different should be ~O(1)
            if ( diff_abs >= 1e-2 )
                return 10 * sigma;
            return 0.0;
        }

        return diff_abs / h;
    };

    // determine smoothness
    const auto is_smooth = [=]( const double diff ) -> bool {
        return diff <= sigma;
    };

    // crop extremes
    // input parameters:
    //      srcs: complete source value view
    //      stncl: local stencil, which might be larger
    //      nn: actual stencil size
    //      tar: bad target intermediate solution
    // output parameter:
    //      tar: fixed value
    // return
    //      true if the value is actually been cropped, false otherwise
    auto crop_extremes = []( const cview_t &srcs, const stencil_t &stncl,
                             const int nn, double &tar ) -> bool {
        double src_max = std::numeric_limits<double>::min();
        double src_min = std::numeric_limits<double>::max();

        for ( int i = 0; i < nn; ++i )
        {
            const double v = srcs[stncl[i]];
            src_max = std::max( src_max, v );
            src_min = std::min( src_min, v );
        }

        // safe guard
        if ( tar >= src_min && tar <= src_max )
            return false;

        // cropping
        if ( tar >= src_max )
            tar = src_max;
        else
            tar = src_min;
        return true;
    };

#ifdef TUNING_INDICATOR_VALUES
    static int _ctr_ = 0;
    std::string fn;
    bool do_output = false;
    if ( d_file_name != "" )
    {
        fn = d_file_name + std::to_string( _ctr_ );
        do_output = true;
        _ctr_++;
    }

    Teuchos::RCP<std::ofstream> file;
    if ( do_output )
        file = Teuchos::rcp( new std::ofstream( fn.c_str() ) );
    const bool can_output = do_output && file->is_open();
#endif

    // actual detection and resolving here

    LO disc_counts = 0;
    const int nv_tar = rangeIntrmV.getLocalLength();
    const int col = domainDistV.numCols();
    // get the stencil size
    Teuchos::ArrayRCP<SupportId> stncl_size = d_pairings->childrenPerParent();
    // get h
    const Teuchos::Array<double> &hs = d_pairings->hs();

    // NOTE here we assume the local index aligns with the global ordering
    // in multivector, which should be fine??

    for ( int dim = 0; dim < col; ++dim )
    {
        // create constant view of src values on this dimension
        cview_t src_view = cview_t( domainDistV[dim], domainDistV.numRows() );
        // create non-constant view of target solution on this dimension
        Teuchos::ArrayRCP<double> tgt_view = rangeIntrmV.getDataNonConst( dim );

        for ( int i = 0; i < nv_tar; ++i )
        {
            // if there is not support, do nothing
            if ( 0 < d_pairings->childCenterIds( i ).size() )
            {
                // get the stencil
                stencil_t stncl = d_pairings->childCenterIds( i );
                // get the actual number of points used in stencil
                const int nn = stncl_size[i];
                // get the h
                const double h = hs[i];
                // get the hh
                const double hh = d_pairings->parentSupportRadius( i );
                // compute the smoothness value
                const double diff = compute_indicator_value(
                    src_view, stncl, nn, tgt_view[i], h, hh );

#ifdef TUNING_INDICATOR_VALUES
                // this format can be easily loaded into Python/MATLAB/Octave
                if ( can_output )
                    *file << dim << ' ' << diff << '\n';
#endif

                if ( is_smooth( diff ) )
                    continue;
                // got one disc
                disc_counts +=
                    crop_extremes( src_view, stncl, nn, tgt_view[i] );
            }
        }
    }

#ifdef TUNING_INDICATOR_VALUES
    if ( can_output )
        file->close();
#endif

    return disc_counts;
}

// added by QC

//---------------------------------------------------------------------------//

} // end namespace DataTransferKit

//---------------------------------------------------------------------------//

#endif // end DTK_MOVINGLEASTSQUARERECONSTRUCTIONOPERATOR_IMPL_HPP

//---------------------------------------------------------------------------//
// end DTK_MovingLeastSquareReconstructionOperator_impl.hpp
//---------------------------------------------------------------------------//
