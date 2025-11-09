def main():
    import sys
    from utils.cli.cli_argument_manager import CLIArgumentManager

    cli = CLIArgumentManager()
    args = cli.parse_arguments()

    ok, errs = cli.validate_arguments(args)
    if not ok:
        for e in errs:
            print(f"❌ {e}", file=sys.stderr)
        return 2

    # Quick direct switches for “management” flags that should act immediately
    if getattr(args, "validate_environment", False):
        cli.validate_environment()
        return 0
    if getattr(args, "list_templates", False):
        cli.list_templates()
        return 0
    if getattr(args, "update_config", None):
        cli.update_config(args.update_config)
        return 0
    if getattr(args, "workflow_status", False) or getattr(args, "status", False) or getattr(args, "list_steps", False) or getattr(args, "validate_config", False):
        ops = {
            "workflow_status": getattr(args, "workflow_status", False),
            "status": getattr(args, "status", False),
            "list_steps": getattr(args, "list_steps", False),
            "validate_config": getattr(args, "validate_config", False),
        }
        cli.print_status_information(symfluence_instance=None, operations=ops)
        return 0
    if getattr(args, "clean", False):
        level = getattr(args, "clean_level", "intermediate")
        cli.clean_workflow_files(level, symfluence_instance=None, dry_run=getattr(args, "dry_run", False))
        return 0
    if getattr(args, "example_notebook", None):
        return cli.launch_example_notebook(args.example_notebook)

    # For installer/binaries
    if getattr(args, "get_executables", None) is not None or getattr(args, "validate_binaries", False) or getattr(args, "force_install", False):
        plan = cli.get_execution_plan(args)
        ok = cli.handle_binary_management(plan)
        return 0 if ok else 1

    # Pour-point setup
    if getattr(args, "pour_point", None):
        plan = cli.get_execution_plan(args)
        pp = plan.get("pour_point", {})
        cli.setup_pour_point_workflow(
            coordinates=pp.get("coordinates"),
            domain_def_method=pp.get("domain_definition_method"),
            domain_name=pp.get("domain_name"),
            bounding_box_coords=pp.get("bounding_box_coords"),
            symfluence_code_dir=None,
        )
        return 0

    # SLURM submit path
    if getattr(args, "submit_job", False):
        plan = cli.get_execution_plan(args)
        cli.handle_slurm_job_submission(plan)
        return 0

    # Default: workflow / steps
    plan = cli.get_execution_plan(args)
    if getattr(args, "dry_run", False):
        print("🔍 DRY RUN — planned execution:")
        print(plan)
        return 0

    # TODO: Call our orchestrator here to execute the plan.
    print("🚀 Execution plan (orchestrator hook point):")
    print(plan)
    return 0
