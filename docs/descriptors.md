# Instance descriptors

Route B uses one generated identity descriptor per ICIR query. The modification is supplied to the
generator only to select the intended subject in the source image; it must not appear in the final
descriptor or describe the requested future state.

The generation instruction enforces these rules:

- select one visible source instance;
- do not use filenames, paths, identifiers, or external category hints;
- do not mention the modification or imagined result;
- describe only visually grounded identity cues;
- avoid background details unless required for disambiguation;
- return captions without reasoning or analysis.

The released table stores the selected second descriptor line as `line2_instance_descriptor`.
Missing lines use, in order, a descriptor from the same query image and a descriptor from the same
instance. Evaluation must record the fallback source. Instance names are not used as model input;
an instance-name fallback is retained only as a detectable data error and should fail strict
reproduction validation.
