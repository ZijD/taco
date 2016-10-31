#include "iterators.h"

#include <iostream>

#include "var.h"
#include "format.h"
#include "internal_tensor.h"
#include "ir.h"
#include "error.h"
#include "util/collections.h"

using namespace std;
using namespace taco::ir;
using taco::internal::Tensor;

namespace taco {
namespace lower {

// class Iterators
Iterators::Iterators(const IterationSchedule& schedule,
                     const map<internal::Tensor,ir::Expr>& tensorVariables) {
  // Create an iterator for each path step
  for (auto& path : schedule.getTensorPaths()) {
    iterators.insert({TensorPathStep(path,-1), storage::Iterator::makeRoot()});

    Tensor tensor = path.getTensor();
    ir::Expr tensorVar = tensorVariables.at(tensor);
    Format format = path.getTensor().getFormat();

    for (size_t i=0; i < path.getSize(); ++i) {
      Level levelFormat = format.getLevels()[i];
      string name = path.getVariables()[i].getName();

      storage::Iterator iterator = storage::Iterator::make(name, tensorVar,
                                                           i, levelFormat);
      iterators.insert({TensorPathStep(path,i), iterator});
    }
  }

  TensorPath resultPath = schedule.getResultTensorPath();
  Tensor tensor = resultPath.getTensor();
  ir::Expr tensorVar = tensorVariables.at(tensor);
  Format format = tensor.getFormat();
  iterators.insert({TensorPathStep(resultPath,-1),
                    storage::Iterator::makeRoot()});
  for (size_t i=0; i < format.getLevels().size(); ++i) {
    taco::Var var = tensor.getIndexVars()[i];
    Level levelFormat = format.getLevels()[i];
    string name = var.getName();
    storage::Iterator resultIterator = storage::Iterator::make(name, tensorVar,
                                                               i, levelFormat);
    resultIterators.insert({var, resultIterator});
    iterators.insert({TensorPathStep(resultPath,i), resultIterator});
  }
}

const storage::Iterator&
Iterators::getIterator(const TensorPathStep& step) const {
  iassert(util::contains(iterators, step));
  return iterators.at(step);
}

const storage::Iterator&
Iterators::getParentIterator(const TensorPathStep& step) const {
  TensorPathStep previousStep(step.getPath(), step.getStep()-1);
  iassert(step.getStep() >= 0);
  iassert(util::contains(iterators, previousStep));
  return iterators.at(previousStep);
}

const storage::Iterator&
Iterators::getResultIterator(const taco::Var& var) const {
  iassert(util::contains(resultIterators, var));
  return resultIterators.at(var);
}

}}