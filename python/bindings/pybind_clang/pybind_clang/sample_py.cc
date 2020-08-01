#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/Comment.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace sample {
namespace {

namespace py = pybind11;
using py_rvp = py::return_value_policy;

static llvm::cl::OptionCategory TestCategory("Test");

/*
Contract:

class MyClass:
    def __init__(...): ...

    def set_ci(self, ci): ...

    def visit_decl(self, decl): ...

    def visit_attr(self, attr): ...

    def visit_stmt(self, stmt): ...

    def visit_expr(self, stmt): ...
*/

// https://clang.llvm.org/docs/RAVFrontendAction.html
// https://clang.llvm.org/docs/LibTooling.html#putting-it-together-the-first-tool

class PyASTVisitor : public clang::RecursiveASTVisitor<PyASTVisitor> {
 public:
  PyASTVisitor(py::object h) : h_(h) {
    visit_decl_ = py::getattr(h_, "visit_decl", py::none());
    visit_attr_ = py::getattr(h_, "visit_attr", py::none());
    visit_stmt_ = py::getattr(h_, "visit_stmt", py::none());
    visit_expr_ = py::getattr(h_, "visit_expr", py::none());
  }

  bool VisitDecl(clang::Decl* d) {
    if (!visit_decl_.is(py::none())) {
      return h_.attr("visit_decl")(d).cast<bool>();
    } else {
      return true;
    }
  }

  bool VisitAttr(clang::Attr* d) {
    if (!visit_attr_.is(py::none())) {
      return h_.attr("visit_attr")(d).cast<bool>();
    } else {
      return true;
    }
  }

  bool VisitStmt(clang::Stmt* d) {
    if (!visit_stmt_.is(py::none())) {
      return h_.attr("visit_decl")(d).cast<bool>();
    } else {
      return true;
    }
  }

  bool VisitDecl(clang::Expr* d) {
    if (!visit_expr_.is(py::none())) {
      return h_.attr("visit_expr")(d).cast<bool>();
    } else {
      return true;
    }
  }

 private:
  py::object h_;
  py::object visit_decl_;
  py::object visit_attr_;
  py::object visit_stmt_;
  py::object visit_expr_;
};

class PyASTConsumer : public clang::ASTConsumer {
 public:
  PyASTConsumer(py::object h) : visitor_(h) {}

  void HandleTranslationUnit(clang::ASTContext &Context) override {
    visitor_.TraverseDecl(Context.getTranslationUnitDecl());
  }

 public:
  PyASTVisitor visitor_;
};

class PyASTFrontendAction : public clang::ASTFrontendAction {
 public:
  PyASTFrontendAction(py::object h) : h_(h) {}

  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(
      clang::CompilerInstance& ci,
      clang::StringRef /* file */) {
    h_.attr("set_ci")(&ci);
    return std::make_unique<PyASTConsumer>(h_);
  }
 private:
  py::object h_;
};

class PyASTFrontendActionFactory
    : public clang::tooling::FrontendActionFactory {
  public:
    PyASTFrontendActionFactory(py::object h) : h_(h) {}

    clang::FrontendAction* create() override {
      return new PyASTFrontendAction(h_);
    }
 private:
  py::object h_;
};

PYBIND11_MODULE(sample, m) {
  {
    // See: clang_indexSourceFile_Impl
    // Should also see: ASTPrinter, CursorVisitor

    using Class = clang::tooling::CommonOptionsParser;
    py::class_<Class>(m, "CommonOptionsParser")
      .def(py::init(
        [](py::list py_argv) {
          // convert :(
          auto cc_argv = py_argv.cast<std::vector<std::string>>();
          int c_argc = cc_argv.size();
          const char** c_argv = new const char*[c_argc];
          for (int i = 0; i < c_argc; ++i) {
            c_argv[i] = cc_argv[i].c_str();
          }
          return new Class(c_argc, c_argv, TestCategory);
        }))
      .def("getCompilations", &Class::getCompilations, py_rvp::reference)
      .def("getSourcePathList", &Class::getSourcePathList);
  }
  {
    using Class = clang::tooling::CompilationDatabase;
    py::class_<Class>(m, "CompilationDatabase");
  }

  {
    using Class = clang::tooling::ClangTool;
    py::class_<Class>(m, "ClangTool")
      .def(py::init(
          [](const clang::tooling::CompilationDatabase& db,
            std::vector<std::string> paths) {
            return new Class(db, paths);
          }))
    .def("run",
        [](Class& self, py::object h) {
          return self.run(new PyASTFrontendActionFactory(h));
        });
  }

  {
    using Class = clang::ASTContext;
    py::class_<Class>(m, "ASTContext");
  }

  {
    using Class = clang::CompilerInstance;
    py::class_<Class>(m, "CompilerInstance")
        .def("getASTContext", &Class::getASTContext, py_rvp::reference);
  }

  {
    using Class = clang::Decl;
    py::class_<Class, std::unique_ptr<Class, py::nodelete>>(m, "Decl")
        .def("dump", [](const Class& self) { self.dump(); })
        .def("getDeclKindName", &Class::getDeclKindName);
  }
  {
    using Class = clang::NamedDecl;
    py::class_<Class, clang::Decl>(m, "NamedDecl")
      .def("getQualifiedNameAsString", &Class::getQualifiedNameAsString);
  }
  {
    using Class = clang::CXXRecordDecl;
    py::class_<Class, clang::NamedDecl>(m, "CXXRecordDecl");
  }
}

}  // namespace
}  // namespace sample
