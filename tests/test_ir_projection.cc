/* Copyright (c) 2022, the adamantine authors.
 *
 * This file is subject to the Modified BSD License and may not be distributed
 * without copyright and license information. Please refer to the file LICENSE
 * for the text and further information on this license.
 */

#include <deal.II/base/mpi.h>
#define BOOST_TEST_MODULE ExperimentaData

#include <Geometry.hh>
#include <experimental_data.hh>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>

#include "main.cc"

namespace utf = boost::unit_test;

BOOST_AUTO_TEST_CASE(project_ray_data_on_mesh)
{
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Mesh from the Tormach wall build
  boost::property_tree::ptree database;
  database.put("import_mesh", false);
  // database.put("length", 99.18e-3);
  // database.put("length_divisions", 29);
  // database.put("height", 21.0e-3);
  // database.put("height_divisions", 6);
  // database.put("width", 99.18e-3);
  // database.put("width_divisions", 29);
  // database.put("material_height", 14.0e-3);

  database.put("length", 400.0e-3);
  database.put("length_divisions", 20);
  database.put("height", 200.0e-3);
  database.put("height_divisions", 10);
  database.put("width", 400.0e-3);
  database.put("width_divisions", 20);
  database.put("material_height", 100.0e-3);

  adamantine::Geometry<3> geometry(communicator, database);
  dealii::parallel::distributed::Triangulation<3> const &tria =
      geometry.get_triangulation();

  dealii::hp::FECollection<3> fe_collection;
  fe_collection.push_back(dealii::FE_Q<3>(1));
  fe_collection.push_back(dealii::FE_Nothing<3>());
  dealii::DoFHandler<3> dof_handler(tria);
  dof_handler.distribute_dofs(fe_collection);

  auto active_cells = 0;
  auto inactive_cells = 0;

  double const material_height = database.get("material_height", 1e9);
  for (auto const &cell :
       dealii::filter_iterators(dof_handler.active_cell_iterators(),
                                dealii::IteratorFilters::LocallyOwnedCell()))
  {
    // If the center of the cell is below material_height, it contains material
    // otherwise it does not.
    if (cell->center()[2] < material_height)
    {
      cell->set_active_fe_index(0);
      active_cells++;
    }
    else
    {
      cell->set_active_fe_index(1);
      inactive_cells++;
    }
  }

  std::cout << "Active cells: " << active_cells
            << " Inactive cells: " << inactive_cells << std::endl;

  // Read the rays from file
  boost::property_tree::ptree experiment_database;
  experiment_database.put("file", "rays_cam-#camera-#frame_test_full.csv");
  experiment_database.put("last_frame", 0);
  experiment_database.put("first_camera_id", 0);
  experiment_database.put("last_camera_id", 0);

  std::cout << "Creating RayTracing object..." << std::endl;
  adamantine::RayTracing ray_tracing(experiment_database);

  // Compute the intersection points
  unsigned int frame = 0;
  std::cout << "Computing intersections..." << std::endl;
  auto points_values = ray_tracing.get_intersection(dof_handler, frame);

  std::cout << "Number of intersections found: " << points_values.points.size()
            << std::endl;

  // Set the values in a temperature vector based on the experimental values
  auto locally_owned_dofs = dof_handler.locally_owned_dofs();
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
  dealii::LinearAlgebra::distributed::Vector<double> temperature(
      locally_owned_dofs, locally_relevant_dofs, communicator);

  std::cout << "Setting field with observations..." << std::endl;
  adamantine::set_with_experimental_data(points_values, dof_handler,
                                         temperature);

  // Create the PostProcessor
  double time = 0.;
  unsigned int cycle = 0;
  std::string filename_prefix = "project_ray_data_on_mesh";

  dealii::DataOut<3> data_out;
  data_out.clear();
  // data_out.attach_dof_handler(dof_handler);
  temperature.update_ghost_values();
  data_out.add_data_vector(dof_handler, temperature, "temperature");

  dealii::types::subdomain_id subdomain_id =
      dof_handler.get_triangulation().locally_owned_subdomain();
  /*
    // Add the subdomain IDs.
    unsigned int const n_active_cells =
        dof_handler.get_triangulation().n_active_cells();
    dealii::types::subdomain_id subdomain_id =
        dof_handler.get_triangulation().locally_owned_subdomain();
    dealii::Vector<float> subdomain(n_active_cells);
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain[i] = subdomain_id;
    data_out.add_data_vector(subdomain, "subdomain");
    */

  // Output the data.
  data_out.build_patches();
  std::string local_filename =
      filename_prefix + "." + dealii::Utilities::int_to_string(subdomain_id, 6);
  std::ofstream output((local_filename + ".vtu").c_str());
  dealii::DataOutBase::VtkFlags flags(time, cycle);
  data_out.set_flags(flags);
  data_out.write_vtu(output);

  // Output the pvtu record.
  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  if (rank == 0)
  {
    std::vector<std::string> filenames;
    unsigned int comm_size =
        dealii::Utilities::MPI::n_mpi_processes(communicator);
    for (unsigned int i = 0; i < comm_size; ++i)
    {
      std::string local_name =
          filename_prefix + "." +
          dealii::Utilities::int_to_string(subdomain_id, 6) + ".vtu";
      filenames.push_back(local_name);
    }
    std::string pvtu_filename =
        filename_prefix + "." +
        dealii::Utilities::int_to_string(subdomain_id, 6) + ".pvtu";
    std::ofstream pvtu_output(pvtu_filename.c_str());
    data_out.write_pvtu_record(pvtu_output, filenames);
  }
}

BOOST_AUTO_TEST_CASE(project_ray_data_on_oversize_mesh)
{
  /*
  MPI_Comm communicator = MPI_COMM_WORLD;

  // Mesh from the Tormach wall build
  boost::property_tree::ptree database;
  database.put("import_mesh", false);
  database.put("length", 400.0e-3);
  database.put("length_divisions", 160);
  database.put("height", 200.0e-3);
  database.put("height_divisions", 80);
  database.put("width", 400.0e-3);
  database.put("width_divisions", 160);
  adamantine::Geometry<3> geometry(communicator, database);
  dealii::parallel::distributed::Triangulation<3> const &tria =
      geometry.get_triangulation();

  dealii::FE_Q<3> fe(1);
  dealii::DoFHandler<3> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Read the rays from file
  boost::property_tree::ptree experiment_database;
  experiment_database.put("file", "rays_cam-#camera-#frame_test_full.csv");
  experiment_database.put("last_frame", 0);
  experiment_database.put("first_camera_id", 0);
  experiment_database.put("last_camera_id", 0);
  adamantine::RayTracing ray_tracing(experiment_database);

  // Compute the intersection points
  unsigned int frame = 0;
  auto points_values = ray_tracing.get_intersection(dof_handler, frame);

  std::cout << "Number of intersections found: " << points_values.points.size()
            << std::endl;

  // Set the values in a temperature vector based on the experimental values
  auto locally_owned_dofs = dof_handler.locally_owned_dofs();
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler,
                                                  locally_relevant_dofs);
  dealii::LinearAlgebra::distributed::Vector<double> temperature(
      locally_owned_dofs, locally_relevant_dofs, communicator);

  adamantine::set_with_experimental_data(points_values, dof_handler,
                                         temperature);

  // Create the PostProcessor
  double time = 0.;
  unsigned int cycle = 0;
  std::string filename_prefix = "project_ray_data_on_oversize_mesh";

  dealii::DataOut<3> data_out;
  data_out.clear();
  data_out.attach_dof_handler(dof_handler);
  temperature.update_ghost_values();
  data_out.add_data_vector(temperature, "temperature");

  // Add the subdomain IDs.
  unsigned int const n_active_cells =
      dof_handler.get_triangulation().n_active_cells();
  dealii::types::subdomain_id subdomain_id =
      dof_handler.get_triangulation().locally_owned_subdomain();
  dealii::Vector<float> subdomain(n_active_cells);
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain[i] = subdomain_id;
  data_out.add_data_vector(subdomain, "subdomain");

  // Output the data.
  data_out.build_patches();
  std::string local_filename =
      filename_prefix + "." + dealii::Utilities::int_to_string(subdomain_id, 6);
  std::ofstream output((local_filename + ".vtu").c_str());
  dealii::DataOutBase::VtkFlags flags(time, cycle);
  data_out.set_flags(flags);
  data_out.write_vtu(output);

  // Output the pvtu record.
  unsigned int rank = dealii::Utilities::MPI::this_mpi_process(communicator);
  if (rank == 0)
  {
    std::vector<std::string> filenames;
    unsigned int comm_size =
        dealii::Utilities::MPI::n_mpi_processes(communicator);
    for (unsigned int i = 0; i < comm_size; ++i)
    {
      std::string local_name =
          filename_prefix + "." +
          dealii::Utilities::int_to_string(subdomain_id, 6) + ".vtu";
      filenames.push_back(local_name);
    }
    std::string pvtu_filename =
        filename_prefix + "." +
        dealii::Utilities::int_to_string(subdomain_id, 6) + ".pvtu";
    std::ofstream pvtu_output(pvtu_filename.c_str());
    data_out.write_pvtu_record(pvtu_output, filenames);
  }
  */
}
