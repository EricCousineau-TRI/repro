// #include <clang/AST/ASTConsumer.h>
// #include <clang/AST/ASTContext.h>
// #include <clang/AST/Comment.h>
// #include <clang/AST/RecursiveASTVisitor.h>
// #include <clang/Basic/Diagnostic.h>
// #include <clang/Basic/SourceLocation.h>
// #include <clang/Frontend/CompilerInstance.h>
// #include <clang/Frontend/FrontendActions.h>
// #include <clang/Tooling/CommonOptionsParser.h>
// #include <clang/Tooling/Tooling.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace sample {
namespace {

namespace py = pybind11;

// static llvm::cl::OptionCategory TestCategory("Test");


// // https://clang.llvm.org/docs/RAVFrontendAction.html
// // https://clang.llvm.org/docs/LibTooling.html#putting-it-together-the-first-tool

// class PyASTVisitor : public clang::RecursiveASTVisitor<Visitor> {
//  public:
//   PyASTVisitor(py::object cls) : cls_(cls) {}
 
//  private:
//   py::object cls_;
// };

// class PyASTConsumer : public clang::ASTConsumer {
//  public:
//   PyASTConsumer(py::object cls) : cls_(cls) {}

//   void HandleTranslationUnit(clang::ASTContext &Context) const {
//     Visitor.TraverseDecl(Context.getTranslationUnitDecl());
//   }

//  public:
//   py::object cls_;
// };

// class PyASTFrontendAction : public clang::ASTFrontendAction {
//  public:
//   PyASTFrontendAction(py::object cls) : cls_(cls)

//   virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
//       clang::CompilerInstance& /* ci */,
//       clang::StringRef /* file */) {
//     return std::make_unique<PyASTConsumer>(cls_);
//   }
//  private:
//   py::object cls_;
// };

// class PyASTFrontendActionFactory : public clang::FrontendActionFactory {
//   public:
//     PyASTFrontendActionFactory(py::object cls) : cls_(cls) {}

//     std::unique_ptr<clang::FrontendAction> create() override {
//       return std::make_unique<PyASTFrontendAction>(cls_);
//     }
//  private:
//   py::object cls_;
// };

PYBIND11_MODULE(sample, m) {
  // {
  //   // See: clang_indexSourceFile_Impl
  //   // Should also see: ASTPrinter, CursorVisitor

  //   using Class = clang::CommonOptionsParser;
  //   py::class_<>(m, "CommonOptionParser")
  //     .def(py::init(
  //       [](py::list py_argv) {
  //         // convert :(
  //         auto cc_argv = py_argv.cast<std::vector<std::string>>();
  //         int c_argc = cc_argv.size();
  //         char** c_argv = new char*[c_argc];
  //         for (int i = 0; i < c_argc; ++i) {
  //           c_argv[i] = cc_argv[i].c_str();
  //         }
  //         return new Class(c_argc, c_argv, TestCategory);
  //       }))
  //     .def("getCompilations", &Class::getCompilations)
  //     .def("getSourcePathList", &Class::getSourcePathList);
  // }
  // py::class_<clang::CompilationDatabase>(m, "CompilationDatabase");

  // {
  //   using Class = clang::ClangTool;
  //   py::class_<>(m, "ClangTool")
  //     .def(py::init(
  //         [](const clang::CompilationDatabase& db, std::vector<std::string> paths) {
  //           return new Class(db, paths);
  //         }))
  //   .def("run",
  //       [](Class& self, py::object cls) {
  //         return self.run(std::make_unique<PyASTFrontendActionFactory>(cls));
  //       });
  // }
}

}  // namespace
}  // namespace sample
