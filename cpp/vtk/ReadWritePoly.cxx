#include <vtkPointSource.h>
#include <vtkXMLPolyDataReader.h>
#include <vtkSmartPointer.h>
#include <vtkXMLPolyDataWriter.h>
#include <vtkPolyData.h>

// Write and read a point cloud.
// Derived from VTK examples, VTK/Examples/Cxx/...:
//  * IO/ReadPolyData
//  * IO/WriteVTP
//  * PolyData/PointSource

int main() {
  // Create a point cloud.
  vtkSmartPointer<vtkPointSource> pointSource =
    vtkSmartPointer<vtkPointSource>::New();
  pointSource->SetCenter(0.0, 0.0, 0.0);
  pointSource->SetNumberOfPoints(50);
  pointSource->SetRadius(5.0);
  pointSource->Update();

  // Write the file.
  const char* filename = "test.vtp";
  vtkSmartPointer<vtkXMLPolyDataWriter> writer =  
    vtkSmartPointer<vtkXMLPolyDataWriter>::New();
  writer->SetFileName(filename);

  writer->SetInput(pointSource->GetOutput());
  writer->Write();

  // Read the file.
  vtkSmartPointer<vtkXMLPolyDataReader> reader =
    vtkSmartPointer<vtkXMLPolyDataReader>::New();
  reader->SetFileName(filename);
  reader->Update();

  return 0;
}
