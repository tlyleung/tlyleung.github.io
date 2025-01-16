require 'fileutils'
require 'open3'
require 'securerandom'
require 'tmpdir'

module Jekyll
  class MermaidRenderer < Jekyll::Generator
    safe true

    def generate(site)
      site.pages.each { |page| process_page(page) }
      site.posts.docs.each { |post| process_page(post) }
    end

    def process_page(page)
      content = page.content
      updated_content = content.gsub(/```mermaid(.*?)```/m) do |match|
        mermaid_code = $1.strip
    
        # Render light and dark SVGs
        light_svg = render_mermaid_to_svg(mermaid_code, 'neutral')
        dark_svg = render_mermaid_to_svg(mermaid_code, 'dark')
    
        # Ensure both SVGs are successfully rendered
        if light_svg && dark_svg
          light_svg.gsub!(/class="flowchart"/, 'class="flowchart block dark:hidden"')
          dark_svg.gsub!(/class="flowchart"/, 'class="flowchart hidden dark:block"')
    
          # Combine the light and dark versions
          <<~HTML
            #{light_svg}
            #{dark_svg}
          HTML
        else
          Jekyll.logger.error "Mermaid Render Failed", "Could not render one or both themes for Mermaid diagram."
          match # Return the original Mermaid block as fallback
        end
      end
    
      page.content = updated_content
    end

    def render_mermaid_to_svg(mermaid_code, theme) 
      Dir.mktmpdir do |dir|        
        input_path = File.join(dir, 'diagram.mmd')
        output_path = File.join(dir, 'diagram.svg')

        # Generate a unique ID for each diagram to replace default "my-svg"
        svg_id = "mermaid-#{SecureRandom.hex(4)}"

        File.write(input_path, mermaid_code)

        css_path = File.expand_path('assets/css/mermaid.css', Dir.pwd)
        command = "node puppeteer-config.js && npx mmdc -i #{input_path} -o #{output_path} --svgId #{svg_id} --theme #{theme} --cssFile #{css_path} --backgroundColor transparent --quiet"
        stdout, stderr, status = Open3.capture3(command)

        if status.success?
          Jekyll.logger.info "Mermaid SVG Generated", output_path
          File.read(output_path)
        else
          Jekyll.logger.error "Mermaid Render Error", stderr
          nil
        end
      end
    end
  end
end
