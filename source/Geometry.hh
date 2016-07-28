/* Copyright (c) 2016, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#ifndef _GEOMETRY_HH_
#define _GEOMETRY_HH_

#include <deal.II/distributed/tria.h>
#include <boost/mpi/communicator.hpp>
#include <boost/property_tree/ptree.hpp>

namespace adamantine
{

template <int dim>
class Geometry
{
public:
  Geometry(boost::mpi::communicator const &communicator,
           boost::property_tree::ptree const &database);

  dealii::parallel::distributed::Triangulation<dim> &get_triangulation();

private:
  dealii::parallel::distributed::Triangulation<dim> _triangulation;
};

template <int dim>
inline dealii::parallel::distributed::Triangulation<dim> &
Geometry<dim>::get_triangulation()
{
  return _triangulation;
}
}

#endif